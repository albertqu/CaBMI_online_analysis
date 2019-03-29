__author__ = 'Albert Qu'

import time, serial, nanpy


canon = [nanpy.Tone.NOTE_C4,
                 nanpy.Tone.NOTE_E4,
                 nanpy.Tone.NOTE_G4,
                 nanpy.Tone.NOTE_G3,
                 nanpy.Tone.NOTE_B3,
                 nanpy.Tone.NOTE_D4,
                 nanpy.Tone.NOTE_A3,
                 nanpy.Tone.NOTE_C4,
                 nanpy.Tone.NOTE_E4,
                 nanpy.Tone.NOTE_E3,
                 nanpy.Tone.NOTE_G3,
                 nanpy.Tone.NOTE_B3
        ]


def nanpy():
    import nanpy
    from nanpy import SerialManager
    print('Search for serial manager')
    try:
        connection = SerialManager()
        #a = nanpy.ArduinoApi(connection=connection)
        outPin = 13
        tonePin = 12
        print('Ready to Connect')
        a = nanpy.ArduinoApi(connection=connection)
        print('Connection set up! with {}'.format(connection))
        a.pinMode(outPin, a.OUTPUT)
        print('OUTPUT SET up')
        aTone = nanpy.Tone(pin=tonePin, connection=connection)  # TODO: CHECK COM
        print('Tone SET UP')


        for i, freq in enumerate(canon):
            aTone.play(freq, 1) # PLAY TONE
            print('Played Freq: {}'.format(freq))
            time.sleep(10)
            op = a.HIGH if i % 2 else a.LOW
            a.digitalWrite(tonePin, op) # Write digital 1 or 0 to TTL
            print('OUTPUTTED {}'.format(op))
            time.sleep(1)
    except:
        print('Failed to Connect')


def main2():
    port = '/dev/tty.usbmodem1411'
    baudrate = 38400
    while True:
        try:
            ser = serial.Serial(port, baudrate)
            print('Connected')
            break
        except:
            print('Failed To Connect')

    lags = []
    lags2 = []

    time.sleep(5)
    for i, freq in enumerate(canon):
        print("Frequency {}".format(freq))
        start = time.time()
        ser.write(str.encode(str(freq) + '!'))
        inter = time.time() - start
        lags.append(inter)
        if i % 2 == 0:
            ser.write(b'~')
        print('Played Freq: {}'.format(freq))
        time.sleep(1)
    while ser.in_waiting:
        lags2.append(int(ser.readline()))
    print(lags)
    print(lags2)
    ser.close()


def main():
    port = '/dev/tty.usbmodem1411'
    baudrate = 38400
    while True:
        try:
            ser = serial.Serial(port, baudrate)
            print('Connected')
            break
        except:
            print('Failed To Connect')
    time.sleep(5)

    for freq in canon:
        ser.write(str.encode(str(freq) + '!'))
        print('Played Freq: {}'.format(freq))
        time.sleep(1)
    ser.close()




if __name__ == '__main__':
    main2()

