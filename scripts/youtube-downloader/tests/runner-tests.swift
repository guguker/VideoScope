import Darwin
import Foundation

@main
struct RunnerTests {
    static func main() {
        var failures = [String]()

        func check(_ condition: @autoclosure () -> Bool, _ message: String) {
            if !condition() { failures.append(message) }
        }

        func run(_ script: String, cancelOn: String? = nil) -> (Int32?, [String]) {
            let runner = CommandRunner()
            var lines = [String]()
            var status: Int32?
            let deadline = Date().addingTimeInterval(10)
            do {
                try runner.start(
                    command: DownloadCommand(
                        executable: "/bin/sh",
                        arguments: ["-c", script],
                        environment: ProcessInfo.processInfo.environment
                    ),
                    onLine: { line in
                        check(Thread.isMainThread, "Output callback must use the main thread")
                        check(status == nil, "Output must arrive before completion")
                        lines.append(line)
                        if let marker = cancelOn, line == marker {
                            runner.cancel()
                            runner.cancel()
                        }
                    },
                    onFinish: { code in
                        check(Thread.isMainThread, "Completion must use the main thread")
                        check(status == nil, "Completion must only run once")
                        status = code
                    }
                )
            } catch {
                failures.append("Unexpected launch failure: \(error)")
            }
            while status == nil && Date() < deadline {
                RunLoop.main.run(until: Date().addingTimeInterval(0.02))
            }
            if status == nil {
                runner.cancel()
                failures.append("Process did not finish within ten seconds")
            }
            return (status, lines)
        }

        // The first Cyrillic character arrives in separate writes, with no newline
        // on the final output. The runner must frame bytes before decoding UTF-8.
        let unicode = run("printf '\\320'; /bin/sleep 0.05; printf '\\237ривет\\n'; printf 'stderr\\n' >&2; printf 'хвост'; exit 7")
        check(unicode.0 == 7, "Nonzero exit status must be preserved")
        check(unicode.1 == ["Привет", "stderr", "хвост"], "Merged Unicode output and final unterminated line must be preserved")

        let carriageReturns = run("printf 'one\\rtwo\\r\\nthree\\n'")
        check(carriageReturns.0 == 0, "Successful process must report zero")
        check(carriageReturns.1 == ["one", "two", "three"], "Progress carriage returns and CRLF must be framed correctly")

        let longLine = run("/usr/bin/awk 'BEGIN { for (i = 0; i < 200000; i++) printf \"x\"; printf \"\\nend\\n\" }'")
        check(longLine.0 == 0, "Large output must drain without deadlock")
        check(longLine.1.count == 2, "An oversized line must remain one logical line")
        check((longLine.1.first?.utf8.count ?? 0) < 70000, "Output line memory must be bounded")
        check(longLine.1.last == "end", "Output must resume after an oversized line")

        // Child ignores SIGINT but acknowledges SIGTERM. The parent ignores both
        // and waits forever, proving the final SIGKILL and group-wide delivery.
        let cancellation = run("""
        trap '' INT
        /bin/sh -c 'trap "echo child-terminated; exit 0" TERM; echo child-ready; while :; do /bin/sleep 0.1; done' &
        trap '' TERM
        while :; do /bin/sleep 10; done
        """, cancelOn: "child-ready")
        check(cancellation.0 == 128 + SIGKILL, "Cancellation must eventually kill an unresponsive parent")
        check(cancellation.1.contains("child-terminated"), "Cancellation must reach child processes in the spawned group; received \(cancellation.1)")

        do {
            let runner = CommandRunner()
            try runner.start(
                command: DownloadCommand(executable: "/nonexistent/videoscope-runner-test", arguments: [], environment: [:]),
                onLine: { _ in failures.append("Failed launch must not produce output callbacks") },
                onFinish: { _ in failures.append("Failed launch must not produce completion callbacks") }
            )
            failures.append("Missing executable must throw")
        } catch {
            // Expected; no subprocess exists and descriptors must be closed.
        }

        if failures.isEmpty {
            print("CommandRunner: all offline tests passed")
        } else {
            failures.forEach { fputs("FAIL: \($0)\n", stderr) }
            exit(1)
        }
    }
}
