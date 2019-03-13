import nanpy
import time
a = nanpy.ArduinoApi()
a.pinMode(10, a.OUTPUT)
aTone = nanpy.Tone(pin=11, connection='COM11')  # TODO: CHECK COM

canon = [aTone.NOTE_C4,
         aTone.NOTE_E4,
         aTone.NOTE_G4,
         aTone.NOTE_G3,
         aTone.NOTE_B3,
         aTone.NOTE_D4,
         aTone.NOTE_A3,
         aTone.NOTE_C4,
         aTone.NOTE_E4,
         aTone.NOTE_E3,
         aTone.NOTE_G3,
         aTone.NOTE_B3
]
for i, freq in enumerate(canon):
    aTone.play(freq, 1) # PLAY TONE
    a.digitalWrite(10, i % 2) # Write digital 1 or 0 to TTL
    time.sleep(0.9)

