import nanpy
a = nanpy.ArduinoApi()
a.pinMode(10, a.OUTPUT)
aTone = nanpy.Tone(pin=11, connection='COM11')  # TODO: CHECK COM
aTone.play(freq, 1) # PLAY TONE
a.digitalWrite(10, 1) # Write digital 1 to TTL
