import os
from datetime import datetime, timedelta
import sys
import glob
import numpy as np
from collections import OrderedDict

# satdir='/arcdata/goes_restricted/grb/goes18/'
# satdir='/arcdata/goes/grb/goes17/'
# satdir='/arcdata/nongoes/japan/himawari08/' #2020_08/2020_08_07_220/0600/'
# satdir='/arcdata/nongoes/meteosat/meteosat11/' #need permissions
# satdir='/apollo/awg_imagery/mtg_prelaunch/' #temporary

sector = "RadF"  # RadC RadM1 RadM2 RadF / JP FLDK / no sector for meteosat?

# Set the datapattern
# goes
datapattern = (
    f"/arcdata/goes/grb/goes16/%Y/%Y_%m_%d_%j/abi/L1b/{sector}/*C13*_s%Y%j%H%M*nc"
)
# ahi
# datapattern = f"/arcdata/nongoes/japan/himawari09/%Y/%Y_%m_%d_%j/%H%M/*B13*{sector}*"
# MTG (prelaunch)
# datapattern = f"/apollo/awg_imagery/mtg_prelaunch/*DEV_%Y%m%d%H%M*0001.nc" #should have hh or hhm (NOT hhmm)
# MSG ; need permissions
# datapattern = f"/arcdata/nongoes/meteosat/meteosat10/%Y/%Y_%m_%d_%j/*EPI*%Y%m%d%H%M*" #keying on epilog file

# Set the output filename
ofilename = "fl.txt"

# Set the start and end datetimes
startdt = dt = datetime(2021, 11, 10, 18, 0)
enddt = datetime(2021, 11, 10, 22, 0)
assert startdt <= enddt

files = []


while startdt <= dt <= enddt:
    c13files = glob.glob(dt.strftime(datapattern))

    if len(c13files):
        #        print(c13files)
        #        sys.exit()
        if "goes1" in datapattern:  # abi
            files.append(c13files[0])

        elif "himawari" in datapattern:  # ahi
            for ff in np.sort(c13files):
                if sector == "FLDK":
                    if ff.endswith("S1010.DAT"):
                        files.append(ff)  # just need one of the 10 files
                elif sector == "JP":
                    files.append(ff)
                else:
                    print(f"sector {sector} is not supported.")
                    sys.exit()
        #                basefile = os.path.basename(ff)
        #                dirname = os.path.dirname(ff)
        #                parts = basefile.split("_")
        #                if sector == "JP":
        #                    if parts[5] == "JP01":
        #                        td = timedelta(minutes=0)
        #                    elif parts[5] == "JP02":
        #                        td = timedelta(minutes=2.5)
        #                    elif parts[5] == "JP03":
        #                        td = timedelta(minutes=5)
        #                    elif parts[5] == "JP04":
        #                        td = timedelta(minutes=7.5)
        #                    else:
        #                        print("parts[5] is invalid: " + parts[5])
        #                        sys.exit()
        #                    filedt = datetime.strptime(parts[2] + parts[3], "%Y%m%d%H%M") + td
        #                    dts.append(filedt.strftime("%Y%m%d-%H%M%S"))
        #                elif sector == "FLDK":
        #                    tmplist = glob.glob(
        #                        dirname + "/" + ("_").join(parts[0:7]) + "*.DAT"
        #                    )
        #                    if len(tmplist) == 10:
        #                        filedt = datetime.strptime(parts[2] + parts[3], "%Y%m%d%H%M")
        #                        dts.append(filedt.strftime("%Y%m%d-%H%M%S"))
        #                        break  # 10 "chunks" per scan in FLDK
        #                    else:
        #                        filedt = datetime.strptime(parts[2] + parts[3], "%Y%m%d%H%M")
        #                        print(f"only {len(tmplist)}/10 files for {filedt}")
        #                else:
        #                    print(f"sector {sector} is not supported.")

        elif "mtg_prelaunch" in datapattern:  # MTG
            for ff in np.sort(c13files):
                basefile = os.path.basename(ff)
                dtpart = basefile.split("DEV_")[1].split("_")[0]
                tmpdt = datetime.strptime(dtpart, "%Y%m%d%H%M%S")
                dts.append(tmpdt.strftime("%Y%m%d-%H%M%S"))

        else:  # meteosat
            for ff in np.sort(c13files):
                basefile = os.path.basename(ff)
                dtpart = basefile.split("EPI______")[1]
                tmpdt = datetime.strptime(dtpart[1:13], "%Y%m%d%H%M")
                dts.append(tmpdt.strftime("%Y%m%d-%H%M%S"))

    # update dt
    dt = dt + timedelta(minutes=1)


ofile = open(ofilename, "w")
for ff in range(len(files)):
    ofile.write(files[ff] + "\n")
ofile.close()


#  elif('mtg_prelaunch' in satdir): #MTG
#    tmplist = np.sort(glob.glob(dt.strftime(f"{satdir}/*DEV_%Y%m%d{hhmm_patterns[dd]}*0001.nc")))
#    for ff in tmplist:
#      basefile = os.path.basename(ff)
#      dtpart = basefile.split('DEV_')[1].split('_')[0]
#      dt = datetime.strptime(dtpart,'%Y%m%d%H%M%S')
#      dts.append(dt.strftime('%Y%m%d-%H%M%S'))
