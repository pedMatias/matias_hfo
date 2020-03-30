### Space Features (continuous values):
- Bottom Right corner: (0.8, 0.9) coordinates;
- Bottom Left corner: (-0.8, 0.9) coordinates;
- Top left corner: (-0.8 -0.9) coordinates;
- Top right corner: (0.8, -0.9) coordinates;
- Mid Field: (0, 0) coordinates;

### Directions (continuous values):
- N: -0.5
- NW: -0.32530558
- W: 0 
- SW:  0.2625805
- S: 0.5
- SE:  0.6974999
- E: 0.99
- NE: -0.6530611

### Can kick? (discrete values):
- -1: Do not have ball
- 1: Has Ball

### Distance to goal (continuous values) [-1, 1]:
- -1: in the goal;
- ex mid far from goal: 0.14;
- ex NW:  0.5957885;

### Distance to opponent (continuous values) [-1, 1]:
- -1: in the goal;
- ex mid far from goalie: 0.14;
- ex NW far from goalie:  0.5;

### Goal Oppening angle [-1, 1]:
É uma metrica um bocado manhosa though. O agente precia basicamente de estar
 dentro da baliza para que seja maior que zero;
- -1: closed angle;
- 1: mid of goal, basically inside the goal;
