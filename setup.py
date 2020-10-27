from distutils.core import setup

setup(
    name = "QCD Data Analysis Tools",
    version = "0.1",
    description = "Analysis tools for Lattice QCD data",
    long_description = """ Ready to go analysis functions and methods to evaluate results from Lattice QCD simulations (and related fields) """,
    author = "Marcel Rodekamp",
    author_email = "marcel.rodekamp@gmail.com",
    package_dir = {
        "qcdanalysistools": "src/"},
    packages = [
        "qcdanalysistools"
    ],
#    scripts=[],
)