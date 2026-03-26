#include "ClearCore.h"
#include <stdint.h>
#include <stdlib.h>

#define ioPort ConnectorUsb
#define ioPortBaudRate 115200
#define MAX_RPM 1000
#define PULSES_PER_REV 2000
#define COOLDOWN_MS 100

// one revolution is ~7.6cm
#define PULSES_PER_CM 263.157894737
#define MAX_GOAL_MOVE_CM 23

// container for the char stream to be read-in
#define IN_BUFFER_LEN 32

// motor connectors
MotorDriver *const motors[MOTOR_CON_CNT] = {
  &ConnectorM0, &ConnectorM1, &ConnectorM2, &ConnectorM3
};

void moveDistance(int m_index, int distance);
void goalieKick(int m_index);

uint32_t lastMotorAction[MOTOR_CON_CNT] = {0,0,0,0};

// Must be less than or equal to clock speed of 100KHz
// Velocity/PPR = RPM
uint32_t velocityLimit  = (MAX_RPM * PULSES_PER_REV) / 60; // pulses/sec
uint32_t accelerationLimit = 300000; // pulses/sec^2

uint8_t cmd_running[4] = {0};

void setup()
{

  ioPort.Mode(Connector::USB_CDC);  
  ioPort.Speed(ioPortBaudRate);
  ioPort.PortOpen();
  while (!ioPort);

  MotorMgr.MotorInputClocking(MotorManager::CLOCK_RATE_LOW);

  // Sets all motor connectors into step and direction mode.
  MotorMgr.MotorModeSet(MotorManager::MOTOR_ALL,
                        Connector::CPM_MODE_STEP_AND_DIR);

  // These lines may be uncommented to invert the output signals of the 
  // Enable, Direction, and HLFB lines. Some motors may have input polarities 
  // that are inverted from the ClearCore's polarity.
  //motor.PolarityInvertSDEnable(true);
  //motor.PolarityInvertSDDirection(true);
  //motor.PolarityInvertSDHlfb(true);

  for(int m = 0; m < MOTOR_CON_CNT; m++)
  {

    // Sets the maximum velocity for each move
    motors[m]->VelMax(velocityLimit);


    // Set the maximum acceleration for each move
    motors[m]->AccelMax(accelerationLimit);

    motors[m]->EnableRequest(true);
  }

}

void loop()
{

  uint32_t i;

  while(1)
  {

    ioPort.SendLine("IDLE... WAITING FOR S");

    char start_input[IN_BUFFER_LEN+1];
    i = 0;
    while(i<IN_BUFFER_LEN)
    {
      if(ioPort.CharPeek() == -1) continue;
      char c = ioPort.CharGet();
      if(c == '\n') break;
      start_input[i++] = c;
    }
    start_input[i] = '\0';

    if(start_input[0] == 'S')
    {
      break;
    }

    delay(250);
  }

  ioPort.SendLine("STARTING");

  char input[IN_BUFFER_LEN+1];
  int cmd_idx = 0;
  uint8_t cmd_ready = 0;

  while(1)
  {

    cmd_ready = 0;
    cmd_idx = 0;

    if(ioPort.CharPeek() != -1)
    {
      while(cmd_idx<IN_BUFFER_LEN) 
      {
        if(ioPort.CharPeek() == -1) continue; 
        char c = ioPort.CharGet(); 
        if(c == '\n') break; 
        input[cmd_idx++] = c; 
      }
      input[cmd_idx] = '\0';
      cmd_ready = 1;
    }

    if(cmd_ready && input[0] == 'Z')
    {
      ioPort.SendLine("STOPPING");
      break;
    }

    // GOALIE KICK
    if(cmd_running[0])
    {
      uint32_t curr_time = Milliseconds();
      if(motors[0]->StepsComplete() && curr_time - lastMotorAction[0] > COOLDOWN_MS)
      {
        cmd_running[0] = 0;
        ioPort.SendLine("ad");
      }
    }
    else if(cmd_ready)
    {
      if(cmd_idx > 0 && input[0] == 'a')
      {
        motors[0]->Move(-PULSES_PER_REV);
        ioPort.SendLine("PERFORMING GOALIE KICK");
        cmd_running[0] = 1;
        uint32_t curr_time = Milliseconds();
        lastMotorAction[0] = curr_time;
      }
    }

    // GOALIE MOVE
    if(cmd_running[1])
    {
      uint32_t curr_time = Milliseconds();
      if(motors[1]->StepsComplete() && curr_time - lastMotorAction[1] > COOLDOWN_MS)
      {
        cmd_running[1] = 0;
        ioPort.SendLine("bd");
      }
    }
    else if(cmd_ready)
    {
      if(cmd_idx > 3 && input[0] == 'b' && isdigit(input[1]) && isdigit(input[2]) && isdigit(input[3]))
      {
        int cm_move = atoi(&input[2]);
        int dir = input[1] - '0';
        if(cm_move < MAX_GOAL_MOVE_CM)
        {
          int pulse_move = cm_move * PULSES_PER_CM;
          pulse_move = dir ? -pulse_move : pulse_move;
          motors[1]->Move(pulse_move);
          ioPort.Send("MOVING GOALIE ");
          ioPort.Send(cm_move);
          ioPort.Send(" cm, direction ");
          ioPort.SendLine(dir);
          cmd_running[1] = 1;
          uint32_t curr_time = Milliseconds();
          lastMotorAction[1] = curr_time;
        }
      }
    }

    // DEF KICK
    if(cmd_running[3])
    {
      uint32_t curr_time = Milliseconds();
      if(motors[3]->StepsComplete() && curr_time - lastMotorAction[3] > COOLDOWN_MS)
      {
        cmd_running[3] = 0;
        ioPort.SendLine("cd");
      }
    }
    else if(cmd_ready)
    {
      if(cmd_idx > 0 && input[0] == 'c')
      {
        motors[3]->Move(-PULSES_PER_REV);
        ioPort.SendLine("PERFORMING DEFENDER KICK");
        cmd_running[3] = 1;
        uint32_t curr_time = Milliseconds();
        lastMotorAction[3] = curr_time;
      }
    }

    // DEF MOVE
    if(cmd_running[2])
    {
      uint32_t curr_time = Milliseconds();
      if(motors[2]->StepsComplete() && curr_time - lastMotorAction[2] > COOLDOWN_MS)
      {
        cmd_running[2] = 0;
        ioPort.SendLine("dd");
      }
    }
    else if(cmd_ready)
    {
      if(cmd_idx > 3 && input[0] == 'd' && isdigit(input[1]) && isdigit(input[2]) && isdigit(input[3]))
      {
        int cm_move = atoi(&input[2]);
        int dir = input[1] - '0';
        if(cm_move < MAX_GOAL_MOVE_CM)
        {
          int pulse_move = cm_move * PULSES_PER_CM;
          pulse_move = dir ? -pulse_move : pulse_move;
          motors[2]->Move(pulse_move);
          ioPort.Send("MOVING DEFENDER ");
          ioPort.Send(cm_move);
          ioPort.Send(" cm, direction ");
          ioPort.SendLine(dir);
          cmd_running[2] = 1;
          uint32_t curr_time = Milliseconds();
          lastMotorAction[2] = curr_time;
        }
      }
    }


  }
}

  /**

    switch(input[0])
    {
      case 'a':
      {
        // Perform goalie kick
        goalieKick(0);
        break;
      }

      case 'b':
      {
        // Perform goalie move, motion given in cm
        if(i > 3 && isdigit(input[1]) && isdigit(input[2]) && isdigit(input[3]))
        {
          int cm_move = atoi(&input[2]);
          int dir = input[1] - '0';
          if(cm_move > MAX_GOAL_MOVE_CM) break;
          int pulse_move = cm_move * PULSES_PER_CM;
          pulse_move = dir ? -pulse_move : pulse_move;
          moveDistance(1, pulse_move);
        }

        break;
      }

      case 'c':
      {
        // Perform defender kick
        goalieKick(3);
        break;
      }

      case 'd':
      {
        // Perform defender move, motion given in cm
        if(i > 3 && isdigit(input[1]) && isdigit(input[2]) && isdigit(input[3]))
        {
          int cm_move = atoi(&input[2]);
          int dir = input[1] - '0';
          if(cm_move > MAX_GOAL_MOVE_CM) break;
          int pulse_move = cm_move * PULSES_PER_CM;
          pulse_move = dir ? -pulse_move : pulse_move;
          moveDistance(2, pulse_move);
        }
        
        break;
      }

      case 'Z':
      {
        stop = true;
      }

      default:
        break;
    }

    if(stop == true)
    {
      ioPort.SendLine("STOPPING");
      break;
    }
  }
}

void goalieKick(int m_index)
{
  MotorDriver *motor = motors[m_index];
  uint32_t curr_time = Milliseconds();

  if(motor->StepsComplete() && curr_time - lastMotorAction[m_index] > COOLDOWN_MS)
  {
    motor->Move(-PULSES_PER_REV);
    lastMotorAction[m_index] = curr_time;
    ioPort.Send("PERFORMED KICK ON MOTOR ");
    ioPort.SendLine(m_index);
  }
  else
  {
    ioPort.SendLine("REJECTED");
  }
}

void moveDistance(int m_index, int distance)
{
  MotorDriver *motor = motors[m_index];
  uint32_t curr_time = Milliseconds();

  if(motor->StepsComplete() && curr_time - lastMotorAction[m_index] > COOLDOWN_MS)
  {
    ioPort.Send("MOVING MOTOR ");
    ioPort.Send(m_index);
    ioPort.Send(" ");
    ioPort.Send(distance);
    ioPort.SendLine(" PULSES");
    motor->Move(distance);
    lastMotorAction[m_index] = curr_time;
  }
  else
  {
    ioPort.SendLine("REJECTED");
    delay(500);
  }
}
*/