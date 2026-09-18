import Darwin
import Foundation

/// A single-use subprocess with merged, bounded UTF-8 output and a private
/// process group. Lifecycle and callbacks are serialized on the main queue.
final class CommandRunner {
    private var hasStarted = false
    private var childPID: pid_t?
    private var processSource: DispatchSourceProcess?
    private var didExit = false
    private var didDrain = false
    private var isCancelling = false
    private var cancellationEscalated = false
    private var onLine: ((String) -> Void)?
    private var onFinish: ((Int32) -> Void)?

    func start(
        command: DownloadCommand,
        onLine: @escaping (String) -> Void,
        onFinish: @escaping (Int32) -> Void
    ) throws {
        precondition(Thread.isMainThread, "CommandRunner.start must run on the main thread")
        guard !hasStarted else {
            throw runnerError("Этот процесс уже был запущен.")
        }
        hasStarted = true
        guard !command.executable.isEmpty,
              !([command.executable] + command.arguments).contains(where: { $0.contains("\0") }),
              !command.environment.contains(where: { key, value in
                  key.isEmpty || key.contains("=") || key.contains("\0") || value.contains("\0")
              }) else {
            throw runnerError("Некорректная команда запуска.")
        }

        var descriptors: [Int32] = [-1, -1]
        guard Darwin.pipe(&descriptors) == 0 else { throw posixError(errno) }
        var readFD = descriptors[0]
        var writeFD = descriptors[1]
        var readerOwnsDescriptor = false
        defer {
            if !readerOwnsDescriptor { Darwin.close(readFD) }
            Darwin.close(writeFD)
        }
        // Keep pipe descriptors away from stdin/stdout/stderr even when launched
        // by a GUI without all three standard descriptors open.
        readFD = try moveAboveStandardDescriptors(readFD)
        writeFD = try moveAboveStandardDescriptors(writeFD)

        var actions: posix_spawn_file_actions_t?
        try checkPOSIX(posix_spawn_file_actions_init(&actions))
        defer { posix_spawn_file_actions_destroy(&actions) }
        try checkPOSIX(posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0))
        try checkPOSIX(posix_spawn_file_actions_adddup2(&actions, writeFD, STDOUT_FILENO))
        try checkPOSIX(posix_spawn_file_actions_adddup2(&actions, writeFD, STDERR_FILENO))
        try checkPOSIX(posix_spawn_file_actions_addclose(&actions, readFD))
        try checkPOSIX(posix_spawn_file_actions_addclose(&actions, writeFD))

        var attributes: posix_spawnattr_t?
        try checkPOSIX(posix_spawnattr_init(&attributes))
        defer { posix_spawnattr_destroy(&attributes) }
        // pgroup zero means the child's PID becomes its new process group ID.
        // Never signal the app's inherited process group.
        try checkPOSIX(posix_spawnattr_setpgroup(&attributes, 0))
        var emptyMask = sigset_t()
        sigemptyset(&emptyMask)
        try checkPOSIX(posix_spawnattr_setsigmask(&attributes, &emptyMask))
        var defaults = sigset_t()
        sigemptyset(&defaults)
        for signal in [SIGINT, SIGTERM, SIGPIPE] { sigaddset(&defaults, signal) }
        try checkPOSIX(posix_spawnattr_setsigdefault(&attributes, &defaults))
        let flags = Int16(POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF)
        try checkPOSIX(posix_spawnattr_setflags(&attributes, flags))

        var arguments = try cStrings([command.executable] + command.arguments)
        defer { arguments.forEach { free($0) } }
        var environment = try cStrings(command.environment.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" })
        defer { environment.forEach { free($0) } }
        var pid: pid_t = 0
        let result = posix_spawn(&pid, command.executable, &actions, &attributes, &arguments, &environment)
        try checkPOSIX(result)

        self.childPID = pid
        self.onLine = onLine
        self.onFinish = onFinish
        let source = DispatchSource.makeProcessSource(identifier: pid, eventMask: .exit, queue: .main)
        source.setEventHandler {
            self.didExit = true
            self.finishIfReady()
        }
        processSource = source
        source.resume()

        readerOwnsDescriptor = true
        let outputFD = readFD
        DispatchQueue.global(qos: .utility).async {
            self.drainOutput(outputFD)
        }
    }

    func cancel() {
        guard Thread.isMainThread else {
            DispatchQueue.main.async { self.cancel() }
            return
        }
        guard childPID != nil, !isCancelling else { return }
        isCancelling = true
        signalGroup(SIGINT)
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            guard self.childPID != nil, self.isCancelling else { return }
            self.signalGroup(SIGTERM)
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
            guard self.childPID != nil, self.isCancelling else { return }
            self.signalGroup(SIGKILL)
            self.cancellationEscalated = true
            self.finishIfReady()
        }
    }

    private func signalGroup(_ signal: Int32) {
        guard let pid = childPID, pid > 1, pid != getpgrp() else { return }
        // The leader remains unreaped until cancellation has fully escalated.
        // Its PID cannot be recycled while delayed group signals are pending.
        _ = Darwin.kill(-pid, signal)
    }

    private func finishIfReady() {
        guard didExit, didDrain, let pid = childPID,
              !isCancelling || cancellationEscalated else { return }
        var status: Int32 = 0
        var result: pid_t
        repeat { result = Darwin.waitpid(pid, &status, WNOHANG) } while result == -1 && errno == EINTR
        if result == 0 {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.01) { self.finishIfReady() }
            return
        }
        childPID = nil
        processSource?.cancel()
        processSource = nil
        let callback = onFinish
        onLine = nil
        onFinish = nil
        let code: Int32
        if result == -1 {
            code = 255
        } else if status & 0x7f == 0 {
            code = (status >> 8) & 0xff
        } else {
            code = 128 + (status & 0x7f)
        }
        callback?(code)
    }

    private func drainOutput(_ descriptor: Int32) {
        var buffer = [UInt8](repeating: 0, count: 8192)
        var framer = OutputLineFramer()
        while true {
            let count = buffer.withUnsafeMutableBytes { Darwin.read(descriptor, $0.baseAddress, $0.count) }
            if count < 0 && errno == EINTR { continue }
            if count <= 0 { break }
            let lines = framer.append(buffer.prefix(count))
            // Synchronous delivery creates bounded backpressure instead of
            // queuing arbitrary amounts of downloader output in main callbacks.
            if !lines.isEmpty {
                DispatchQueue.main.sync { lines.forEach { self.onLine?($0) } }
            }
        }
        Darwin.close(descriptor)
        if let trailing = framer.finish() {
            DispatchQueue.main.sync { self.onLine?(trailing) }
        }
        DispatchQueue.main.async {
            self.didDrain = true
            self.finishIfReady()
        }
    }

    private func moveAboveStandardDescriptors(_ descriptor: Int32) throws -> Int32 {
        if descriptor <= STDERR_FILENO {
            let moved = fcntl(descriptor, F_DUPFD_CLOEXEC, 3)
            guard moved >= 0 else { throw posixError(errno) }
            Darwin.close(descriptor)
            return moved
        }
        guard fcntl(descriptor, F_SETFD, FD_CLOEXEC) == 0 else { throw posixError(errno) }
        return descriptor
    }

    private func cStrings(_ values: [String]) throws -> [UnsafeMutablePointer<CChar>?] {
        var pointers = [UnsafeMutablePointer<CChar>?]()
        for value in values {
            guard let pointer = strdup(value) else {
                pointers.forEach { free($0) }
                throw posixError(ENOMEM)
            }
            pointers.append(pointer)
        }
        pointers.append(nil)
        return pointers
    }

    private func checkPOSIX(_ code: Int32) throws {
        if code != 0 { throw posixError(code) }
    }

    private func posixError(_ code: Int32) -> Error {
        NSError(domain: NSPOSIXErrorDomain, code: Int(code))
    }

    private func runnerError(_ message: String) -> Error {
        NSError(domain: "YouTubeDownloader.CommandRunner", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
}

private struct OutputLineFramer {
    private let maximumLineBytes = 64 * 1024
    private var bytes = [UInt8]()
    private var truncated = false
    private var previousWasCarriageReturn = false

    mutating func append(_ incoming: ArraySlice<UInt8>) -> [String] {
        var lines = [String]()
        for byte in incoming {
            if byte == 10 && previousWasCarriageReturn {
                previousWasCarriageReturn = false
                continue
            }
            previousWasCarriageReturn = byte == 13
            if byte == 10 || byte == 13 {
                lines.append(takeLine())
            } else if bytes.count < maximumLineBytes {
                bytes.append(byte)
            } else {
                truncated = true
            }
        }
        return lines
    }

    mutating func finish() -> String? {
        guard !bytes.isEmpty || truncated else { return nil }
        return takeLine()
    }

    private mutating func takeLine() -> String {
        var text: String?
        if truncated {
            // A byte limit can bisect a code point. Remove at most its three
            // trailing bytes before falling back to replacement for invalid UTF-8.
            for removed in 0...3 {
                text = String(bytes: bytes.dropLast(removed), encoding: .utf8)
                if text != nil { break }
            }
        }
        let line = (text ?? String(decoding: bytes, as: UTF8.self)) + (truncated ? " … [строка сокращена]" : "")
        bytes.removeAll(keepingCapacity: true)
        truncated = false
        return line
    }
}
