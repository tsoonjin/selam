(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("socreport" "fypca")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("enumitem" "shortlabels") ("inputenc" "utf8")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "url")
   (TeX-run-style-hooks
    "latex2e"
    "chap0_abstract"
    "socreport"
    "socreport10"
    "fullpage"
    "float"
    "hyperref"
    "graphicx"
    "enumitem"
    "inputenc")
   (LaTeX-add-labels
    "fig:transdec_aerial"
    "fig:robosub2016_tasks"
    "fig:water_surface_effect"
    "fig:vision_challenges"
    "fig:proposed_vision_framework"
    "fig:main_methodology"
    "fig:dataset_methodology"
    "fig:training_methodology"
    "fig:tracking_methodology")
   (LaTeX-add-bibliographies
    "fyp"))
 :latex)

