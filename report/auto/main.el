(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("socreport" "hyp")))
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
    "mathtools"
    "fullpage"
    "float"
    "hyperref"
    "graphicx"
    "amsmath"
    "caption"
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
    "fig:tracking_methodology"
    "fig:colorconstancy_results"
    "fig:color_gamma"
    "fig:grey_pixel"
    "fig:spatial_colorconstancy"
    "fig:underwater_image_enhancement_results"
    "fig:fusion_pipeline"
    "fig:illumination_compensation_results"
    "fig:light_compensation"
    "fig:proposal_results"
    "fig:mser_proposal"
    "fig:salient_proposal"
    "fig:similar_color"
    "sec:color_descriptors"
    "sec:shape_descriptors"
    "fig:inner_shapecontext"
    "fig:gp"
    "fig:tracker"
    "fig:dataset1"
    "fig:dataset2"
    "table:competing_trackers"
    "table:raw_results")
   (LaTeX-add-bibliographies
    "fyp"))
 :latex)

