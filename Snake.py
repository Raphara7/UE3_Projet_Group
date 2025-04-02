import curses
import random

# Configuration de la fenêtre
curses.initscr()
win = curses.newwin(20, 60, 0, 0)  # Crée une fenêtre de 20x60
win.keypad(1)
curses.noecho()
curses.curs_set(0)
win.border(0)
win.nodelay(1)

# Initialisation des variables
snake = [(4, 10), (4, 9), (4, 8)]  # Corps du serpent
food = (10, 20)  # Position initiale de la nourriture

win.addch(food[0], food[1], '#')  # Affiche la nourriture

# Logique du jeu
score = 0

ESC = 27
key = curses.KEY_RIGHT  # Direction initiale

while key != ESC:
    win.addstr(0, 2, 'Score : ' + str(score) + ' ')
    win.timeout(150 - (len(snake)//5 + len(snake)//10) % 120)  # Augmente la vitesse

    prev_key = key
    event = win.getch()
    key = event if event != -1 else prev_key

    if key not in [curses.KEY_LEFT, curses.KEY_RIGHT, curses.KEY_UP, curses.KEY_DOWN, ESC]:
        key = prev_key

    # Calculer la prochaine position de la tête
    y = snake[0][0]
    x = snake[0][1]
    if key == curses.KEY_DOWN:
        y += 1
    if key == curses.KEY_UP:
        y -= 1
    if key == curses.KEY_LEFT:
        x -= 1
    if key == curses.KEY_RIGHT:
        x += 1

    # Insérer une nouvelle tête
    snake.insert(0, (y, x))

    # Vérifier si la tête est sur la nourriture
    if snake[0] == food:
        score += 1
        food = None
        while food is None:
            nf = (random.randint(1, 18), random.randint(1, 58))
            food = nf if nf not in snake else None
        win.addch(food[0], food[1], '#')
    else:
        # Déplacer le serpent
        last = snake.pop()
        win.addch(last[0], last[1], ' ')

    # Vérifier les collisions avec les murs ou lui-même
    if (y == 0 or y == 19 or x == 0 or x == 59 or snake[0] in snake[1:]):
        break

    win.addch(snake[0][0], snake[0][1], '*')

curses.endwin()
print("\nScore final : ", score)
