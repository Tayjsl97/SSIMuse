# Data

This directory contains only compact, processed piano rolls that can be used
directly by the released scripts.

```text
controlled/POP909/  16-bar controlled-data crops (included)
controlled/Pop1K7/  source and preparation note (data not redistributed)
real/MPDSet29/      29 query/source pairs as aligned melody piano rolls
real/CopyrightCases40/  18 infringement queries and 39 candidate sources
```

The included arrays use `uint8` rather than the original local `float64`
storage. Values are unchanged; every converted array was checked element by
element against the local experiment input.

The repository's MIT license applies to SSIMuse code. Dataset attribution and
upstream licensing information are stated in each dataset directory.
