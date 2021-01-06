from distutils.core import setup

setup(
    name = "QCD Data Analysis Tools",
    version = "0.1",
    description = "Analysis tools for Lattice QCD data",
    long_description = """ Ready to go analysis functions and methods to evaluate results from Lattice QCD simulations (and related fields) """,
    author = "Marcel Rodekamp",
    author_email = "marcel.rodekamp@gmail.com",
    package_dir = {
        "qcdanalysistools": "src/",
            "qcdanalysistools.analysis": "src/StatisticalAnalysis",
                "qcdanalysistools.analysis.Blocking": "src/StatisticalAnalysis/Blocking",
                "qcdanalysistools.analysis.Jackknife": "src/StatisticalAnalysis/Jackknife",
                "qcdanalysistools.analysis.Bootstrap": "src/StatisticalAnalysis/Bootstrap",
            "qcdanalysistools.observables": "src/Observables",
            "qcdanalysistools.stats": "src/Stats",
            "qcdanalysistools.fitting": "src/Fitting",
    },
    packages = [
        "qcdanalysistools",
        "qcdanalysistools.analysis",
            "qcdanalysistools.analysis.Blocking",
            "qcdanalysistools.analysis.Jackknife",
            "qcdanalysistools.analysis.Bootstrap",
        "qcdanalysistools.observables",
        "qcdanalysistools.stats",
        "qcdanalysistools.fitting",
    ],
#    scripts=[],
)
