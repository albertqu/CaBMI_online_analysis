___author__ = 'Albert Qu'

import serial


class ArduinoDelegate:

    def __init__(self, port, baudrate=38400):
        self.ser = serial.Serial(port, baudrate)
        self.rewardKey = '~'
        self.delimiter = '!'

    def reward(self):
        self.ser.write(str.encode(self.rewardKey))

    def playTone(self, freq, stdout=True):
        """ By default duration = 1s, To Modify Duration,
        Change DURATION in client.ino"""
        self.ser.write(str.encode(str(freq) + self.delimiter))
        if stdout:
            print('Tone played at {}Hz'.format(freq))

    def close(self):
        self.ser.close()
        print('Delegate Successfully Closed!')
