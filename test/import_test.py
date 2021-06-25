import qcdanalysistools
print(f"Imported qcdanalysistools: {qcdanalysistools.__file__}")
print(f"Imported Bootstrap: {qcdanalysistools.analysis.Bootstrap.bootstrap}")
print(f"Imported Jackknife: {qcdanalysistools.analysis.Jackknife.jackknife}")
print(f"Imported Blocking: {qcdanalysistools.analysis.Blocking.blocking}")
print(f"Imported Fitting Base: {qcdanalysistools.fitting.FitBase}")

# optional
import qcdanalysistools.fileIO.affIO
print(f"Imported AFF reader: {qcdanalysistools.fileIO.affIO.read_data}")
