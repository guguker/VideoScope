import AppKit

/// A standalone desktop utility. It has no connection to VideoScope indexing.
final class DownloaderApp: NSObject, NSApplicationDelegate, NSWindowDelegate {
    private var window: NSWindow!
    private let root: URL
    private var directory: URL { root.appendingPathComponent("Данные матчи", isDirectory: true) }
    private let links = NSTextView()
    private let quality = NSPopUpButton()
    private let format = NSPopUpButton()
    private let cookies = NSPopUpButton()
    private let subtitles = NSButton(checkboxWithTitle: "Субтитры RU / EN, включая автоматические", target: nil, action: nil)
    private let thumbnail = NSButton(checkboxWithTitle: "Сохранить обложку", target: nil, action: nil)
    private let download = NSButton(title: "Скачать", target: nil, action: nil)
    private let cancel = NSButton(title: "Отменить", target: nil, action: nil)
    private let paste = NSButton(title: "Вставить из буфера", target: nil, action: nil)
    private let clear = NSButton(title: "Очистить", target: nil, action: nil)
    private let status = NSTextField(labelWithString: "Готов к загрузке")
    private let detail = NSTextField(labelWithString: "Вставьте одну или несколько ссылок — каждую с новой строки.")
    private let formatHint = NSTextField(wrappingLabelWithString: "")
    private let progress = NSProgressIndicator()
    private let log = NSTextView()
    private var currentRunner: CommandRunner?
    private var queue: [String] = []
    private var position = 0
    private var succeeded = 0
    private var failed = 0
    private var savedFiles: Set<String> = []
    private var stopped = false
    private var quitting = false
    private var settings = DownloadSettings()
    private var tools: DownloadTools?
    private let heights: [Int?] = [nil, 4320, 2160, 1440, 1080, 720, 480, 360, 240, 144]
    private let formats: [OutputFormat] = [.mp4, .mkv, .m4a, .mp3]
    private let browsers: [String?] = [nil, "arc", "chrome", "safari", "firefox"]

    init(root: URL) {
        self.root = root
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        buildMenu()
        buildWindow()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        window.makeFirstResponder(links)
    }

    private func buildMenu() {
        let menu = NSMenu()
        let appItem = NSMenuItem()
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Завершить YouTube Downloader", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        menu.addItem(appItem)
        let edit = NSMenuItem()
        edit.title = "Правка"
        let editMenu = NSMenu(title: "Правка")
        editMenu.addItem(withTitle: "Отменить ввод", action: Selector(("undo:")), keyEquivalent: "z")
        editMenu.addItem(.separator())
        editMenu.addItem(withTitle: "Вырезать", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        editMenu.addItem(withTitle: "Копировать", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: "Вставить", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editMenu.addItem(withTitle: "Выделить всё", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        edit.submenu = editMenu
        menu.addItem(edit)
        NSApp.mainMenu = menu
    }

    private func label(_ text: String, size: CGFloat = 13, weight: NSFont.Weight = .regular) -> NSTextField {
        let view = NSTextField(labelWithString: text)
        view.font = .systemFont(ofSize: size, weight: weight)
        return view
    }

    private func horizontal(_ views: [NSView], spacing: CGFloat = 10) -> NSStackView {
        let stack = NSStackView(views: views)
        stack.orientation = .horizontal
        stack.alignment = .centerY
        stack.spacing = spacing
        return stack
    }

    private func spacer() -> NSView {
        let view = NSView()
        view.setContentHuggingPriority(.defaultLow, for: .horizontal)
        return view
    }

    private func field(_ name: String, control: NSView) -> NSStackView {
        let stack = NSStackView(views: [label(name, weight: .medium), control])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 6
        control.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        return stack
    }

    private func textBox(_ textView: NSTextView, height: CGFloat, editable: Bool) -> NSScrollView {
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.borderType = .bezelBorder
        scroll.autohidesScrollers = true
        textView.isEditable = editable
        textView.isSelectable = true
        textView.isRichText = false
        textView.importsGraphics = false
        textView.allowsUndo = editable
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticSpellingCorrectionEnabled = false
        textView.font = editable ? .systemFont(ofSize: 13) : .monospacedSystemFont(ofSize: 11, weight: .regular)
        textView.textColor = .labelColor
        textView.backgroundColor = .textBackgroundColor
        textView.textContainerInset = NSSize(width: 10, height: 9)
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.containerSize = NSSize(width: 700, height: CGFloat.greatestFiniteMagnitude)
        scroll.documentView = textView
        scroll.heightAnchor.constraint(equalToConstant: height).isActive = true
        return scroll
    }

    private func buildWindow() {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 820, height: 790),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "YouTube → Данные матчи"
        window.minSize = NSSize(width: 780, height: 800)
        window.isReleasedWhenClosed = false
        window.delegate = self
        window.center()

        let content = NSStackView()
        content.orientation = .vertical
        content.alignment = .leading
        content.spacing = 14
        content.translatesAutoresizingMaskIntoConstraints = false
        window.contentView!.addSubview(content)
        NSLayoutConstraint.activate([
            content.leadingAnchor.constraint(equalTo: window.contentView!.leadingAnchor, constant: 24),
            content.trailingAnchor.constraint(equalTo: window.contentView!.trailingAnchor, constant: -24),
            content.topAnchor.constraint(equalTo: window.contentView!.topAnchor, constant: 22),
            content.bottomAnchor.constraint(lessThanOrEqualTo: window.contentView!.bottomAnchor, constant: -22)
        ])
        func add(_ view: NSView) {
            content.addArrangedSubview(view)
            view.widthAnchor.constraint(equalTo: content.widthAnchor).isActive = true
        }

        add(label("Скачать с YouTube", size: 25, weight: .bold))
        let description = label("Видео и аудио сохраняются в папку проекта «Данные матчи».")
        description.textColor = .secondaryLabelColor
        add(description)

        paste.target = self
        paste.action = #selector(pasteLinks)
        clear.target = self
        clear.action = #selector(clearLinks)
        add(horizontal([label("Ссылки на видео", weight: .semibold), spacer(), paste, clear]))
        add(textBox(links, height: 110, editable: true))

        quality.addItems(withTitles: ["Лучшее доступное", "До 8K · 4320p", "До 4K · 2160p", "До 1440p", "До 1080p", "До 720p", "До 480p", "До 360p", "До 240p", "До 144p"])
        quality.selectItem(at: 4)
        format.addItems(withTitles: ["Видео · MP4", "Видео · MKV", "Аудио · M4A", "Аудио · MP3"])
        format.target = self
        format.action = #selector(formatChanged)
        cookies.addItems(withTitles: ["Без cookies", "Arc", "Chrome", "Safari", "Firefox"])
        cookies.toolTip = "Если YouTube просит войти: выберите браузер, где вы уже вошли. Cookies читаются только при запуске загрузки с этим выбором."
        let options = horizontal([field("Разрешение", control: quality), field("Формат", control: format), field("Вход через браузер", control: cookies)], spacing: 16)
        options.distribution = .fillEqually
        add(options)
        formatHint.font = .systemFont(ofSize: 11)
        formatHint.textColor = .secondaryLabelColor
        formatHint.heightAnchor.constraint(equalToConstant: 32).isActive = true
        add(formatHint)
        formatChanged()
        add(horizontal([subtitles, spacer(), thumbnail]))

        let folder = NSTextField(labelWithString: directory.path)
        folder.isSelectable = true
        folder.lineBreakMode = .byTruncatingMiddle
        folder.font = .systemFont(ofSize: 11)
        folder.textColor = .secondaryLabelColor
        folder.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        folder.toolTip = directory.path
        let openFolder = NSButton(title: "Открыть папку", target: self, action: #selector(revealFolder))
        add(horizontal([folder, openFolder]))

        download.bezelStyle = .rounded
        download.controlSize = .large
        download.keyEquivalent = "\r"
        download.target = self
        download.action = #selector(startDownloads)
        cancel.bezelStyle = .rounded
        cancel.target = self
        cancel.action = #selector(cancelDownloads)
        cancel.isEnabled = false
        add(horizontal([download, cancel, spacer(), label("По одной ссылке · до 50 видео", size: 11)]))
        status.font = .systemFont(ofSize: 13, weight: .semibold)
        add(status)
        progress.style = .bar
        progress.isIndeterminate = false
        progress.minValue = 0
        progress.maxValue = 100
        add(progress)
        detail.font = .systemFont(ofSize: 11)
        detail.textColor = .secondaryLabelColor
        detail.lineBreakMode = .byTruncatingMiddle
        add(detail)
        add(textBox(log, height: 140, editable: false))
    }

    @objc private func formatChanged() {
        let audio = format.indexOfSelectedItem >= 2
        quality.isEnabled = !audio && currentRunner == nil
        if audio {
            formatHint.stringValue = "Только звуковая дорожка. Разрешение видео не используется."
        } else if format.indexOfSelectedItem == 0 {
            formatHint.stringValue = "Выбранное разрешение — верхний предел. MP4 в 4K/8K может требовать плеер с AV1/VP9; совместимость с QuickTime зависит от кодека."
        } else {
            formatHint.stringValue = "Лучшее видео со звуком в пределах выбранного разрешения. MKV сохраняет исходные кодеки без перекодирования."
        }
    }

    @objc private func pasteLinks() {
        guard let text = NSPasteboard.general.string(forType: .string), !text.isEmpty else { return }
        links.string += (links.string.isEmpty || links.string.hasSuffix("\n") ? "" : "\n") + text
    }

    @objc private func clearLinks() { links.string = "" }

    @objc private func revealFolder() {
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            NSWorkspace.shared.open(directory)
        } catch { showError(error.localizedDescription) }
    }

    private func showError(_ message: String) {
        let alert = NSAlert()
        alert.messageText = "Не удалось начать загрузку"
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.beginSheetModal(for: window)
    }

    @objc private func startDownloads() {
        guard currentRunner == nil else { return }
        do {
            queue = try validatedVideoURLs(links.string)
            tools = try DownloadTools.discover(preferredDirectory: root.appendingPathComponent(".youtube-downloader/runtime/bin"))
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            showError(error.localizedDescription)
            return
        }
        settings = DownloadSettings()
        settings.height = heights[quality.indexOfSelectedItem]
        settings.format = formats[format.indexOfSelectedItem]
        settings.subtitles = subtitles.state == .on
        settings.thumbnail = thumbnail.state == .on
        settings.browser = browsers[cookies.indexOfSelectedItem]
        position = 0
        succeeded = 0
        failed = 0
        stopped = false
        log.string = ""
        setBusy(true)
        appendLog("Папка: \(directory.path)")
        appendLog("В очереди: \(queue.count). Формат: \(settings.format.rawValue.uppercased()).")
        startNext()
    }

    private func setBusy(_ busy: Bool) {
        download.isEnabled = !busy
        cancel.isEnabled = busy
        paste.isEnabled = !busy
        clear.isEnabled = !busy
        links.isEditable = !busy
        format.isEnabled = !busy
        quality.isEnabled = !busy && format.indexOfSelectedItem < 2
        cookies.isEnabled = !busy
        subtitles.isEnabled = !busy
        thumbnail.isEnabled = !busy
    }

    private func startNext() {
        guard !stopped, position < queue.count, let tools else {
            finishQueue()
            return
        }
        let url = queue[position]
        savedFiles = []
        status.stringValue = "Видео \(position + 1) из \(queue.count) · получение данных…"
        detail.stringValue = url
        progress.isIndeterminate = true
        progress.startAnimation(nil)
        appendLog("\n[\(position + 1)/\(queue.count)] \(url)")
        do {
            let command = try makeDownloadCommand(url: url, settings: settings, directory: directory, tools: tools)
            let runner = CommandRunner()
            currentRunner = runner
            try runner.start(command: command, onLine: { [weak self] line in
                self?.consume(line)
            }, onFinish: { [weak self] code in
                self?.completed(code: code)
            })
        } catch {
            currentRunner = nil
            appendLog("Ошибка запуска: \(error.localizedDescription)")
            completed(code: 1)
        }
    }

    private func consume(_ line: String) {
        if line.hasPrefix("VS_PROGRESS|") {
            let fields = line.components(separatedBy: "|")
            if fields.count >= 4 {
                let value = fields[1].replacingOccurrences(of: "%", with: "").trimmingCharacters(in: .whitespaces)
                if let percent = Double(value), percent.isFinite {
                    progress.stopAnimation(nil)
                    progress.isIndeterminate = false
                    progress.doubleValue = max(0, min(percent, 100))
                }
                if !stopped {
                    status.stringValue = "Видео \(position + 1) из \(queue.count) · скачивание"
                    detail.stringValue = "Текущая дорожка: \(fields[1]) · \(fields[2]) · осталось \(fields[3])"
                }
            }
        } else if line.hasPrefix("VS_SAVED|") {
            let path = String(line.dropFirst("VS_SAVED|".count))
            let saved = URL(fileURLWithPath: path).standardizedFileURL
            let expected = directory.standardizedFileURL.path + "/"
            if saved.path.hasPrefix(expected),
               let attributes = try? FileManager.default.attributesOfItem(atPath: saved.path),
               attributes[.type] as? FileAttributeType == .typeRegular,
               let size = attributes[.size] as? NSNumber, size.int64Value > 0 {
                savedFiles.insert(saved.path)
                appendLog("Сохранено: \(saved.lastPathComponent)")
            } else {
                appendLog("Не удалось подтвердить сохранённый файл: \(path)")
            }
        } else {
            appendLog(line)
            if !stopped && (line.hasPrefix("[Merger]") || line.hasPrefix("[VideoRemuxer]") || line.hasPrefix("[ExtractAudio]")) {
                status.stringValue = "Видео \(position + 1) из \(queue.count) · обработка файла…"
                detail.stringValue = "Объединение дорожек или конвертация. Дождитесь сообщения о сохранении."
                progress.isIndeterminate = true
                progress.startAnimation(nil)
            }
        }
    }

    private func completed(code: Int32) {
        currentRunner = nil
        progress.stopAnimation(nil)
        progress.isIndeterminate = false
        if stopped {
            appendLog("Загрузка остановлена. Частичные файлы сохранены для повторной попытки.")
            finishQueue()
            return
        }
        if code == 0 && !savedFiles.isEmpty {
            succeeded += 1
            appendLog("Готово.")
        } else {
            failed += 1
            appendLog(code == 0 ? "Файл не получен: видео пропущено или недоступно для выбранных параметров." : "Загрузка завершилась с ошибкой (код \(code)). Подробности выше.")
        }
        position += 1
        // Yield to AppKit between jobs so Cancel/close remains responsive.
        DispatchQueue.main.async { [weak self] in self?.startNext() }
    }

    private func finishQueue() {
        setBusy(false)
        progress.stopAnimation(nil)
        progress.isIndeterminate = false
        progress.doubleValue = !stopped && failed == 0 && succeeded > 0 ? 100 : 0
        status.stringValue = "\(stopped ? "Остановлено" : "Готово") · успешно: \(succeeded), ошибок: \(failed)"
        detail.stringValue = stopped
            ? "Повторная загрузка продолжит .part-файлы, если YouTube поддерживает докачку."
            : (failed > 0 ? "Причина ошибки указана в журнале. Если YouTube просит вход, выберите браузер выше." : "Файлы доступны по кнопке «Открыть папку».")
        if quitting { NSApp.reply(toApplicationShouldTerminate: true) }
    }

    @objc private func cancelDownloads() {
        guard !stopped else { return }
        stopped = true
        cancel.isEnabled = false
        status.stringValue = "Останавливаю загрузку…"
        if let currentRunner { currentRunner.cancel() }
    }

    private func appendLog(_ line: String) {
        guard !line.isEmpty else { return }
        let storage = log.textStorage!
        let attributes: [NSAttributedString.Key: Any] = [.font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular), .foregroundColor: NSColor.labelColor]
        storage.append(NSAttributedString(string: line + "\n", attributes: attributes))
        if storage.length > 100_000 {
            storage.deleteCharacters(in: NSRange(location: 0, length: storage.length - 80_000))
        }
        log.scrollToEndOfDocument(nil)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard currentRunner != nil else { return .terminateNow }
        quitting = true
        cancelDownloads()
        return .terminateLater
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        if currentRunner != nil {
            NSApp.terminate(nil)
            return false
        }
        return true
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
guard CommandLine.arguments.count == 2,
      FileManager.default.fileExists(atPath: CommandLine.arguments[1] + "/Скачать с YouTube.command") else {
    let alert = NSAlert()
    alert.messageText = "Запустите «Скачать с YouTube.command» из папки проекта."
    alert.runModal()
    exit(1)
}
let delegate = DownloaderApp(root: URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true))
app.delegate = delegate
app.run()
