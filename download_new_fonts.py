import urllib.request
import ssl
from pathlib import Path

ROOT = Path(__file__).resolve().parent
fonts_dir = ROOT / "fonts"
fonts_dir.mkdir(exist_ok=True)

ctx = ssl._create_unverified_context()

fonts_to_download = {
    # 1. Libre Baskerville (Serif)
    "libre_baskerville": [
        ("LibreBaskerville-Variable.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/librebaskerville/LibreBaskerville%5Bwght%5D.ttf")
    ],
    # 2. Merriweather (Serif)
    "merriweather": [
        ("Merriweather-Variable.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/merriweather/Merriweather%5Bopsz%2Cwdth%2Cwght%5D.ttf")
    ],
    # 3. PT Serif (Serif)
    "pt_serif": [
        ("PTSerif-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/ptserif/PT_Serif-Web-Regular.ttf")
    ],
    # 4. Crimson Text (Serif)
    "crimson_text": [
        ("CrimsonText-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/crimsontext/CrimsonText-Regular.ttf"),
        ("CrimsonText-Bold.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/crimsontext/CrimsonText-Bold.ttf"),
        ("CrimsonText-Italic.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/crimsontext/CrimsonText-Italic.ttf")
    ],
    # 5. EB Garamond (Serif)
    "eb_garamond": [
        ("EBGaramond-Variable.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/ebgaramond/EBGaramond%5Bwght%5D.ttf")
    ],
    # 6. Cormorant Garamond (Serif)
    "cormorant_garamond": [
        ("CormorantGaramond-Variable.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf")
    ],
    # 7. Playfair Display (Serif)
    "playfair_display": [
        ("PlayfairDisplay-Variable.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf")
    ],
    # 8. Lato (Sans)
    "lato": [
        ("Lato-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/lato/Lato-Regular.ttf"),
        ("Lato-Bold.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/lato/Lato-Bold.ttf"),
        ("Lato-Italic.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/lato/Lato-Italic.ttf")
    ],
    # 9. Source Sans 3 (Sans)
    "source_sans": [
        ("SourceSans3-Variable.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/sourcesans3/SourceSans3%5Bwght%5D.ttf")
    ],
    # 10. PT Sans (Sans)
    "pt_sans": [
        ("PTSans-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/ptsans/PT_Sans-Web-Regular.ttf")
    ],
    # 11. Arimo (Sans)
    "arimo": [
        ("Arimo-Variable.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/arimo/Arimo%5Bwght%5D.ttf")
    ],
    # 12. Montserrat (Sans)
    "montserrat": [
        ("Montserrat-Variable.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf")
    ],
    # 13. Inter (Sans)
    "inter": [
        ("Inter-Variable.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf")
    ],
    # 14. Raleway (Sans)
    "raleway": [
        ("Raleway-Variable.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/raleway/Raleway%5Bwght%5D.ttf")
    ],
    # 15. Nunito (Sans)
    "nunito": [
        ("Nunito-Variable.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/nunito/Nunito%5Bwght%5D.ttf")
    ],
    # 16. Special Elite (Typewriter)
    "special_elite": [
        ("SpecialElite-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/apache/specialelite/SpecialElite-Regular.ttf")
    ],
    # 17. VT323 (Dot-matrix / Terminal)
    "vt323": [
        ("VT323-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/vt323/VT323-Regular.ttf")
    ],
    # 18. Cousine (Monospace)
    "cousine": [
        ("Cousine-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/cousine/Cousine-Regular.ttf")
    ],
    # 19. Space Mono (Monospace)
    "space_mono": [
        ("SpaceMono-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/spacemono/SpaceMono-Regular.ttf"),
        ("SpaceMono-Bold.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/spacemono/SpaceMono-Bold.ttf")
    ],
    # 20. Caveat (Handwriting)
    "caveat": [
        ("Caveat-Variable.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/caveat/Caveat%5Bwght%5D.ttf")
    ],
    # 21. Indie Flower (Handwriting)
    "indie_flower": [
        ("IndieFlower-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/indieflower/IndieFlower-Regular.ttf")
    ],
    # 22. Shadows Into Light (Handwriting)
    "shadows_into_light": [
        ("ShadowsIntoLight-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/shadowsintolight/ShadowsIntoLight.ttf")
    ]
}

downloaded = {}
for fam, items in fonts_to_download.items():
    downloaded[fam] = []
    for fname, url in items:
        out_path = fonts_dir / fname
        if not out_path.is_file() or out_path.stat().st_size == 0:
            print(f"Downloading {fam}: {fname}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                data = resp.read()
                out_path.write_bytes(data)
                print(f"Saved {fname} ({len(data):,} bytes)")
        downloaded[fam].append(fname)

print("All new fonts downloaded successfully!")
