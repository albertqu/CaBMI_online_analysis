// AUTHOR: Albert Qu, 03/14/2019
#define AUDIO_PIN  11
#define TTL_PIN 10
#define TONE_DURATION 1000 // tone duration in ms
#define LED_DURATION 100 // LED duration in ms
String inString = "";
int start;
int lag;
void setup()
{
  Serial.begin(38400);  // initialize serial communications at 9600 bps
  pinMode(TTL_PIN, OUTPUT);
  digitalWrite(TTL_PIN, LOW);
}
// Cite: [Arduino - StringToIntExample](https://www.arduino.cc/en/Tutorial.StringToIntExample)
void loop()
{ 
  if (Serial.available()>0) { 
    // Available
    int inChar = Serial.read(); // convert the incoming byte to a char and add it to the string:
    if (inChar == '~') {
      // Reward Effected
      digitalWrite(TTL_PIN, HIGH);
      Serial.println(1);
      delay(LED_DURATION);
      digitalWrite(TTL_PIN, LOW); 
    }
    else if (isDigit(inChar)) {
      // Frequency input!
      /*if (inString == "") { Uncomment to measure System Lag
        start = micros();
      }*/
      inString += (char)inChar;
    }
    
    else if (inChar == '!') {
      // Use '!' as delimiter, frequency input ended
      tone(AUDIO_PIN, inString.toInt(), TONE_DURATION); 
      /* Uncomment to measure system lag
      lag = (int) (micros()-start);
      Serial.println(lag);
      //delay(1000); //No Delay In Process Wanted*/
      // clear the string for new input:
      inString = "";
    }
  }
}
