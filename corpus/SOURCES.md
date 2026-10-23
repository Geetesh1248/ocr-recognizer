# Corpus Sources and Documentation

The text data used for synthetic text recognition training is compiled from public-domain literary works and synthetically generated structured form fields. No external models or OCR APIs were used.

## 1. Project Gutenberg Public Domain Texts
- **Alice's Adventures in Wonderland** (Lewis Carroll)
  - URL: `https://www.gutenberg.org/ebooks/11`
  - License: Public Domain (US)
  - File: `corpus/gutenberg_alice.txt`
- **Pride and Prejudice** (Jane Austen)
  - URL: `https://www.gutenberg.org/ebooks/1342`
  - License: Public Domain (US)
  - File: `corpus/gutenberg_pride.txt`
- **The Adventures of Sherlock Holmes** (Arthur Conan Doyle)
  - URL: `https://www.gutenberg.org/ebooks/1661`
  - License: Public Domain (US)
  - File: `corpus/gutenberg_sherlock.txt`

## 2. Form & Document Entities
- **File**: `corpus/form_data.json`
- **Contents**: Common business and legal form field names, header labels, currency amounts, dates, tracking numbers, invoice identifiers, and address components typical of scanned form documents such as FUNSD.
- **Coverage**: Handcrafted to ensure dense coverage of numerals, symbols (`$`, `%`, `@`, `&`, `#`, `*`, `+`, `=`, `/`, `\`, `[]`, `()`, `{}`), punctuation, dates, codes, and uppercase words.
