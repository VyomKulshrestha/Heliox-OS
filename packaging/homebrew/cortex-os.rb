cask "cortex-os" do
  version "0.2.0"
  sha256 :no_check # Will be filled after release

  url "https://github.com/VyomKulshrestha/Cortex-OS/releases/download/v#{version}/Cortex-OS_#{version}_aarch64.dmg",
      verified: "github.com/VyomKulshrestha/Cortex-OS/"

  name "Cortex-OS"
  desc "AI System Control Agent — control your computer with voice, text, and gestures"
  homepage "https://github.com/VyomKulshrestha/Cortex-OS"

  livecheck do
    url :url
    strategy :github_latest
  end

  depends_on formula: "python@3.12"

  app "Cortex-OS.app"

  zap trash: [
    "~/.config/cortex-os",
    "~/Library/Application Support/com.cortexos.app",
  ]
end
