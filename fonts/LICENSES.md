# Font Licenses and Attribution

All font files bundled in this directory are openly licensed for free commercial and non-commercial research, modification, and redistribution under the SIL Open Font License (OFL-1.1), Apache 2.0, or permissive public domain notices.

## 1. DejaVu Fonts
- **Files**: `DejaVuSans*.ttf`, `DejaVuSerif*.ttf`, `DejaVuSansMono*.ttf` (12 core files)
- **License**: Bitstream Vera Fonts Copyright / DejaVu Fonts License (Permissive Open License)
- **Source**: http://dejavu-fonts.org/ / bundled with Matplotlib
- **Notice**:
  Fonts are based on Bitstream Vera Fonts. Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved.
  DejaVu changes are in the public domain.

## 2. Computer Modern Fonts
- **Files**: `cmtt10.ttf`
- **License**: Public Domain / Knuth TeX Font License
- **Source**: Matplotlib font library

## 3. STIX Fonts
- **Files**: `STIXGeneral.ttf`, `STIXGeneralBol.ttf`, `STIXGeneralItalic.ttf`, `STIXGeneralBolIta.ttf`
- **License**: SIL Open Font License, Version 1.1 (OFL-1.1)
- **Source**: Scientific and Technical Information Exchange (STIX) Fonts Project

## 4. Google Fonts & Open Source Typefaces (OFL-1.1 / Apache-2.0)
The following fonts are licensed under the SIL Open Font License, Version 1.1 (OFL-1.1) or Apache License, Version 2.0:
- **Arimo** (`Arimo-Variable.ttf`): Apache 2.0, Copyright Steve Matteson.
- **Caveat** (`Caveat-Variable.ttf`): OFL-1.1, Copyright 2014-2021 Caveat Project Authors.
- **Cormorant Garamond** (`CormorantGaramond-Variable.ttf`): OFL-1.1, Copyright Christian Thalmann.
- **Courier Prime** (`CourierPrime-*.ttf`): OFL-1.1, Copyright Quote-Unquote Apps (Alan Dague-Greene).
- **Cousine** (`Cousine-Regular.ttf`): Apache 2.0 / OFL-1.1, Copyright Steve Matteson.
- **Crimson Text** (`CrimsonText-*.ttf`): OFL-1.1, Copyright Sebastian Kosch.
- **Cutive Mono** (`CutiveMono-Regular.ttf`): OFL-1.1, Copyright Vernon Adams.
- **EB Garamond** (`EBGaramond-Variable.ttf`): OFL-1.1, Copyright Georg Duffner.
- **Fira Mono** (`FiraMono-Regular.ttf`): OFL-1.1, Copyright Carrois Corporate GbR and Edenspiekermann AG.
- **Inconsolata** (`Inconsolata-Variable.ttf`): OFL-1.1, Copyright Raph Levien.
- **Indie Flower** (`IndieFlower-Regular.ttf`): OFL-1.1, Copyright Kimberly Geswein.
- **Inter** (`Inter-Variable.ttf`): OFL-1.1, Copyright Rasmus Andersson.
- **Lato** (`Lato-*.ttf`): OFL-1.1, Copyright Lukasz Dziedzic.
- **Libre Baskerville** (`LibreBaskerville-Variable.ttf`): OFL-1.1, Copyright Impallari Type.
- **Lora** (`Lora-Variable.ttf`): OFL-1.1, Copyright Cyreal (Olga Karpushina, Alexei Vanyashin).
- **Merriweather** (`Merriweather-Variable.ttf`): OFL-1.1, Copyright Sorkin Type Co.
- **Montserrat** (`Montserrat-Variable.ttf`): OFL-1.1, Copyright Julieta Ulanovsky.
- **Noto Sans & Noto Serif** (`NotoSans-Variable.ttf`, `NotoSerif-Variable.ttf`): OFL-1.1, Copyright The Noto Project Authors.
- **Nunito** (`Nunito-Variable.ttf`): OFL-1.1, Copyright Vernon Adams.
- **Open Sans** (`OpenSans-Variable.ttf`): OFL-1.1, Copyright The Open Sans Project Authors.
- **Playfair Display** (`PlayfairDisplay-Variable.ttf`): OFL-1.1, Copyright Claus Eggers Sorensen.
- **PT Sans & PT Serif** (`PTSans-Regular.ttf`, `PTSerif-Regular.ttf`): Paratype Free Font License (OFL compatible).
- **Raleway** (`Raleway-Variable.ttf`): OFL-1.1, Copyright Matt McInerney, Pablo Impallari, Rodrigo Fuenzalida.
- **Roboto** (`Roboto-Variable.ttf`): Apache-2.0, Copyright Google Inc.
- **Shadows Into Light** (`ShadowsIntoLight-Regular.ttf`): OFL-1.1, Copyright Kimberly Geswein.
- **Source Code Pro** (`SourceCodePro-Variable.ttf`): OFL-1.1, Copyright Adobe Systems Incorporated.
- **Source Sans 3** (`SourceSans3-Variable.ttf`): OFL-1.1, Copyright Adobe Systems Incorporated.
- **Space Mono** (`SpaceMono-*.ttf`): OFL-1.1, Copyright Colophon Foundry.
- **Special Elite** (`SpecialElite-Regular.ttf`): Apache-2.0, Copyright Astigmatic (AOETI).
- **VT323** (`VT323-Regular.ttf`): OFL-1.1, Copyright Peter Hull.

---

## Train / Validation Split Policy

To rigorously evaluate generalisation on unseen fonts, font families are split strictly by **Family**:
- **Training Families (32 families)**: DejaVu Sans, DejaVu Serif, DejaVu Sans Mono, Computer Modern Mono, STIX, Roboto, Open Sans, Courier Prime, Cutive Mono, Inconsolata, Libre Baskerville, Merriweather, PT Serif, Crimson Text, EB Garamond, Cormorant Garamond, Playfair Display, Lato, Source Sans, PT Sans, Arimo, Montserrat, Inter, Raleway, Nunito, Special Elite, VT323, Cousine, Space Mono, Caveat, Indie Flower, Shadows Into Light.
- **Validation Families (5 held-out families)**: Lora, Source Code Pro, Noto Sans, Noto Serif, Fira Mono.

Intersection: **Zero shared families**.
Assignments are defined in `fonts/splits.json`.
