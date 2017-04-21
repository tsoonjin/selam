(TeX-add-style-hook
 "slides"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "beamer"
    "beamer10"
    "pgfpages"
    "amsmath"
    "amssymb"
    "enumerate"
    "epsfig"
    "bbm"
    "calc"
    "color"
    "ifthen"
    "capt-of"))
 :latex)

