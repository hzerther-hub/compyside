# macOS bundle notes

To finalise a `.dmg` from this repo you need:

* `icon.icns` — multi-resolution (16/32/64/128/256/512/1024) `icns` bundle. Drop
  it next to `Info.plist`. Until it exists, Briefcase will fall back to its
  default scaffold icon.
* Apple Developer ID Application certificate installed in the Keychain.
* `xcrun notarytool` credentials stored via
  `xcrun notarytool store-credentials <profile>`.

## Build & package

```bash
briefcase create macOS
briefcase build macOS
briefcase package macOS --identity "Developer ID Application: Your Name" \
                       --entitlements packaging/macOS/entitlements.plist
briefcase package macOS --no-notarize           # for unsigned dev builds
briefcase run macOS                             # smoke-test locally
```

The signed + notarised `.dmg` lands in `dist/Compositor-0.1.0.dmg`.
