Das Projekt "Gesture Rex Adventure" konzentriert sich auf die Steuerung eines Computerspiels mit Hilfe von Computer Vision.
In der Web-App sind drei verschiedene Ansätze für die Gestenkontrolle eingebettet.

Gruppenmitglieder:

  Jannis Trapp (3017015)  
  Jessika Keil (4681952)  
  Johanna Stahl (1419076)

![image](https://github.com/user-attachments/assets/a6d0e7d2-427d-4846-9f66-775435c47411)

Starten der WebApp:
1.  Erforderliche Bibliotheken installieren. Diese befinden sich im Ordner der WebApp unter Requirements.txt.
    Dies kann bspw. durch folgenden Befehl geschehen:
    "pip install flask opencv-python-headless pyautogui numpy mediapipe tensorflow"
2.  mit einem Terminal zum WebApp-Ordner navigieren.
3.  Den Befehl "python app.py" ausführen. Nun warten, bis die Module korrekt importiert wurden.
4.  Nun kann die WebApp über: http://127.0.0.1:5000/ geöffnet werden.
5.  Auf der Startseite kann der gewünschte Algorithmus nun ausgewählt und ausprobiert werden.

Ein Demo-Video zur Verwendung der WebApp ist sowohl in der Präsentation, als auch im Abgabeordner enthalten.

Außerdem befindet sich im Abgabeordner neben dem Ordner "WebApp" auch noch eine Datei "Modelltraining.jpyn".
Diese enhält den Code, der zum Training des eigenen Modells (Variante 3) verwendet wurde.