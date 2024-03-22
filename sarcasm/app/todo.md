# 12/09/2022 Parameterbereiche für GUI

## currently working on:
    pseudo transactions when storing structure/metadata/roidata,
    is implemented,
    commit calls need to be done in gui


## Todo's

- ~~Bei Parameters Import/Export sollte es einen “Set default” Button geben, der alle parameter auf default zurücksetzt~~
- ~~Für Structure und Motion Teile sollte es oben einen Button “Analyze Structure”/“Analyze Motion” geben, der mit den unten gesetzten Parametern alle Analysen durchführt, ohne dass man die einzelnen Buttons klicken muss.~~ 
- ~~Könntest du für alle Parameter Tooltips implementieren: wenn man mit der Maus darüber geht, erscheint nach einer Sekunde oder so die Beschreibung der Parameter aus den Docstrings. Das wäre großartig!~~
- ~~Die Funktion “Export Data” sollte auch in den Header, also direkt unter Parameter Import/Export.
   Ich habe auch versucht eine csv zu erzeugen, das funktioniert noch nicht. 
   Brauchst du da noch Funktionen von meiner Seite?~~
- ~~Wenn ich ein neues File lade, aber davor bereits eines geladen hatte, verschwindet das alte nicht aus Napari...~~
- Wir benötigen einen Batch Mode: 
- Statt einzelne Files zu laden, sollte man auch eine Liste laden können (z.B. alle tif-files aus einem Ordner und/oder einzelne Files auswählen)
- Dann sollte immer das in Napari angezeigt werden, das gerade ausgewählt ist aus der Liste
- Unten sollten dann zwei Optionen sein: 1. Analyze selected file, 2. Analyze all files
- Wir müssen uns dann noch Gedanken machen, wie man beim Batch mode den Export der Daten gestaltet, da die Daten dann hochdimensional sind. 
- Es wäre super, wenn wir über Napari auch einige der Zwischenergebnisse anzeigen könnten bei der Structure Analysis (Z-band segmentation, Sarcomere lengths, Myofibril lines). Das macht das ganze deutlich verständlicher. Ich habe dafür bereits Funktionen geschrieben bei plots.py. Bei Z-band segmentation eine Darstellung der Z-bands mit Farben ihrer labels, bei sarcomere length und orientation eine Bild mit den Längen/orientierungen als Farbe, und für die Myofibril lines einfach die Linien. Für die Motion Analyse könnte eventuell ein Fenster aufpoppen das die Resultate anzeigt?
- Ich habe einen kleinen Fix gepusht für den Fall dass “pixelsize” oder “tint” existiert aber None ist, das ist immer der Fall wenn man diese Metadaten nicht aus den Tifs extrahieren kann.


## Solved Todo's Testing:


