# Navn: Reading in data
# Oprettet: 25-05-2022
# Senest ændret: 25-05-2022
# Data ligger i onedrive konto SataDtorage@gmail.com, pass: bethlamamifr

################### CHANGELOG ###########################
# FZC: Created the document                             #
#########################################################

################### DESCRIPTION #########################
# This program is reading in the data for the           #
# final project in Applied Machine Learning             #
#########################################################

# 1. Reading in packages

import pandas as pd
import os as os
# os.chdir("LOKAL STI TIL DREVAMPPEN") # Til dem der skal have ændret deres current directory

# Jeg (Michael) har ikke sat current directory til google drive mappen, så der skal jeg lige navigere hen (MichaelDir)
MichaelDir = str("Final project/data")
standardDir = ""

# Hvis du har sat current directory til google mappen brug standardDir
UsedDir = MichaelDir
df = pd.read_csv(UsedDir + '/SwissData/SwissData2.txt')


