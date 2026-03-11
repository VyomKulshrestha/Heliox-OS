cask "heliox-os" do
  version "0.2.0"
  sha256 :no_check # Will be filled after release

  url "https://github.com/VyomKulshrestha/Heliox-OS/releases/download/v#{version}/Heliox OS_#{version}_aarch64.dmg",
      verified: "github.com/VyomKulshrestha/Heliox-OS/"

  name "Heliox OS"
  desc "AI System Control Agent — control your computer with voice, text, and gestures"
  homepage "https://github.com/VyomKulshrestha/Heliox-OS"

  livecheck do
    url :url
    strategy :github_latest
  end

  depends_on formula: "python@3.12"

  app "Heliox OS.app"

  zap trash: [
    "~/.config/heliox-os",
    "~/Library/Application Support/com.helioxos.app",
  ]
end
