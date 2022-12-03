import pickle
import random
import sys
import time

import pygame
from pygame.locals import *

BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GREY = (96, 96, 96)
LBLUE = (68, 68, 255)
LRED = (255, 68, 68)
BROWN = (102, 51, 0)


class RenderBoard(object):
    """Class to render the board to a Pygame surface.
    Attributes:
        rows (int): Number of rows in the board.
        columns (int): Number of columns in the board.
        box_width (int): Width of individual *square* box in the board.
        margin_width (int, optional): Width of margins between individual boxes in the board.
        pygame_surface (pygame.Surface): Reference to Pygame surface.
        board_x (int): X placement of the board.
        board_y (int): Y placement of the board.
        board_width (int, optional): Board width.
        board_height (int): Board height.
        box_mapping (dict): int -> colour mapping for the boxes
    """

    def __init__(self, pygame_surface, board_x, board_y, rows, columns, box_mapping, box_width=20, margin_width=5):
        self.rows = rows
        self.columns = columns
        self.box_width = box_width
        self.margin_width = margin_width
        self.pygame_surface = pygame_surface
        self.box_mapping = box_mapping

        # x,y placement of the board
        self.board_x = board_x
        self.board_y = board_y

        # Total width and height of board
        self.board_width = self.columns * self.box_width + (self.columns - 1) * self.margin_width
        self.board_height = self.rows * self.box_width + (self.rows - 1) * self.margin_width

    def draw_board(self, game_board):
        """Draw board to the Pygame surface.
        Args:
            game_board (list): List containing a game board state. len(game_board) == rows * columns.
        """
        assert len(game_board) == self.rows * self.columns
        curr_x = self.board_x
        curr_y = self.board_y

        for y in range(self.rows):
            for x in range(self.columns):
                # Get the box as an offset in the board and then obtain the box mapping
                # this can be a function call or a colour.
                # E.g: board[1] = box1 = 1 -> BLUE, board[2] = box2 = 3 -> foo()
                offset = x + y * self.columns
                box = game_board[offset]
                # Draw a row of boxes
                pygame.draw.rect(self.pygame_surface, self.box_mapping[box],
                                 (curr_x, curr_y, self.box_width, self.box_width))
                curr_x += self.box_width + self.margin_width

            # Start new row
            curr_x = self.board_x
            curr_y += self.box_width + self.margin_width

    def mouse_in_board(self, mouse_x, mouse_y):
        """Check if the mouse coordinates fall within the board.
        Args:
            mouse_x (int): X mouse coordinate.
            mouse_y (int): Y mouse coordinate.
        Returns:
            True if successful, False otherwise.
        """
        if (self.board_x <= mouse_x <= self.board_x + self.board_width) and \
                (self.board_y <= mouse_y <= self.board_y + self.board_height):
            return True

        return False

    def mouse_in_margin(self, mouse_x, mouse_y, box_x, box_y):
        """Determine if the mouse is inside a margin of the board.
        A margin is a "dead" zone of the board.
        Args:
            mouse_x (int): X mouse coordinate.
            mouse_y (int): Y mouse coordinate.
            box_x (int): X location of box in the board.
            box_y (int): Y location of box in the board.
        Returns:
            True if successful, False otherwise.
        """
        # Use get_box_coord() to get box_x and box_y
        if (box_x + 1) * self.box_width + box_x * self.margin_width < (mouse_x - self.board_x) \
                < (box_x + 1) * self.box_width + (box_x + 1) * self.margin_width:
            return True
        if (box_y + 1) * self.box_width + box_y * self.margin_width < (mouse_y - self.board_y) \
                < (box_y + 1) * self.box_width + (box_y + 1) * self.margin_width:
            return True

        return False

    def get_box_coord(self, mouse_x, mouse_y):
        """Determine the X, Y location of a box inside a board from the mouse coordinates.
        Args:
            mouse_x (int): X mouse coordinate.
            mouse_y (int): Y mouse coordinate.
        Returns:
            box_x (int): X location of box in the board.
            box_y (int): Y location of box in the board.
        """
        box_x = (mouse_x - self.board_x) // (self.box_width + self.margin_width)
        box_y = (mouse_y - self.board_y) // (self.box_width + self.margin_width)

        return box_x, box_y

    def draw_box(self, box_x, box_y, colour):
        """Draw an individual box in the board based on box coordinates to the Pygame surface.
        Args:
            box_x (int): X location of box in the board.
            box_y (int): Y location of box in the board.
            colour (tuple): (R,G,B).
        """
        x = box_x * self.box_width + box_x * self.margin_width + self.board_x
        y = box_y * self.box_width + box_y * self.margin_width + self.board_y
        pygame.draw.rect(self.pygame_surface, colour, (x, y, self.box_width, self.box_width))

    def draw_margin_box(self, box_x, box_y, colour):
        """Draw a box spanning the margin in the board based on box coordinates to the Pygame surface.
        One can create a "box highlight" effect by first calling this method followed by draw_box().
        There is an issue with calling draw.rect(... thickness) in that is renders a strange thickness and
        seems to be incorrect.
        Args:
            box_x (int): X location of box in the board.
            box_y (int): Y location of box in the board.
            colour (tuple): (R,G,B).
        """
        x = box_x * self.box_width + box_x * self.margin_width - self.margin_width + self.board_x
        y = box_y * self.box_width + box_y * self.margin_width - self.margin_width + self.board_y
        pygame.draw.rect(self.pygame_surface, colour, (x, y, self.box_width + 2 * self.margin_width,
                                                       self.box_width + 2 * self.margin_width))


class Board(object):
    """Board class for state and move validation.
    Attributes:
        rows (int): Number of rows in the board.
        columns (int): Number of columns in the board.
        __board (list): Private list containing board state.
        __active_player (int): Numerical number of current/active player.
        active_player (property, int): ""
        __inactive_player (int): Numerical number of inactive player.
        inactive_player (property, int): ""
        board_list (property, list): The raw board state as a list, use this for rendering.
        __players_position (dict): Private dict of players position, key corresponds to player.
        player1_pos (property, tuple): (X, Y) location of player 1. Can be None if game just started.
        player2_pos (property, tuple): (X, Y) location of player 2. Can be None if game just started.
    """

    # Board/box states, must be unique and in powers of 2 (bit masking)
    PLAYER1 = 1
    PLAYER2 = 2
    BOX_CLEAR = 4
    BOX_BLOCK = 8
    BOX_BLOCKED_MASK = 16  # Leave blocked as the last entry of block states and the highest

    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns

        self.__board = []
        self.__active_player = Board.PLAYER1
        self.__inactive_player = Board.PLAYER2

        # Player 1 and player 2 coordinates set to None for start of game
        self.__players_position = {Board.PLAYER1: None, Board.PLAYER2: None}
        self.clear_board()

    def __eq__(self, other):
        """Equality operator.
        Args:
            other (Board): The other Board object to compare to.
        """
        return self.__dict__ == other.__dict__

    def clear_board(self):
        """Clear the game board list.
        """
        self.__board = [Board.BOX_CLEAR for _ in range(self.rows * self.columns)]

    def gen_random_blocked_boxes(self, min, max):
        """Generate random blocked boxes.
        Args:
            min (int): Minimum bound of blocked boxes.
            max (int): Maximum bound of blocked boxes.
        """
        # Number of blocked boxes in random range of [min, max]
        num = random.randint(min, max)

        for _ in range(num):
            pos = random.randint(0, self.columns * self.rows - 1)
            self.__board[pos] = Board.BOX_BLOCK | Board.BOX_BLOCKED_MASK

    @property
    def active_player(self):
        return self.__active_player

    @property
    def inactive_player(self):
        return self.__inactive_player

    @property
    def board_list(self):
        return self.__board

    def player_pos(self, player):
        """Obtain the position for "player".
        Args:
            player (int): Player to obtain position for.
        Returns:
            (int, int): Position.
        """
        assert player in self.__players_position
        return self.__players_position[player]

    @property
    def player1_pos(self):
        return self.__players_position[Board.PLAYER1]

    @property
    def player2_pos(self):
        return self.__players_position[Board.PLAYER2]

    def offset(self, x, y):
        """Obtain an offset in to the board list based on X, Y board/box coordinates.
        Args:
            x (int): X board coordinate.
            y (int): Y board coordinate.
        """
        return x + y * self.columns

    def box_blocked(self, x, y):
        """Determine if a box in the board is blocked.
        Args:
            x (int): X box coordinate.
            y (int): Y box coordinate.
        Returns:
            True if successful, False otherwise.
        """
        return (self.__board[self.offset(x, y)] & Board.BOX_BLOCKED_MASK) == Board.BOX_BLOCKED_MASK

    def __block_box(self, x, y, board_state):
        """Block a box and its state in the X, Y coordinates of the board.
        Args:
            x (int): X box coordinate.
            y (int): Y box coordinate.
            board_state (int): State to block, i.e. PLAYER1, PLAYER2 etc.
        """
        self.__board[self.offset(x, y)] = board_state | Board.BOX_BLOCKED_MASK

    def get_free_boxes(self):
        """Return a list containing a tuple of (X, Y) coordinates of all free
        i.e. non-blocked boxes in the board.
        Returns:
            moves (list): List of tuples (X, Y) containing all free boxes.
        """
        moves = []
        for y in range(self.rows):
            for x in range(self.columns):
                if not self.box_blocked(x, y):
                    moves.append((x, y))

        return moves

    def get_legal_moves(self, player=None):
        """Return a list of all legal moves for the a player.
        By default use the active player if player is not set.
        Args:
            player (int): The active or inactive player. Default to the
            active player if player is None.
        Returns:
            moves (list): List of tuples (X, Y) with coordinates of valid moves.
        """
        if player is None:
            loc = self.__players_position[self.__active_player]
        else:
            loc = self.__players_position[player]

        # Game just started so return all free squares as legal moves
        if not loc:
            return self.get_free_boxes()

        # Define the directional deltas which span out in
        # 8 directions from any position in the grid
        dirs_deltas = [(-1, -1), (+0, -1), (+1, -1), (+1, +0),
                       (+1, +1), (+0, +1), (-1, +1), (-1, +0)]

        moves = []
        for dx, dy in dirs_deltas:
            # Explore all possible directional deltas from the starting position
            x, y = loc
            while (0 <= (x + dx) < self.columns) and (0 <= (y + dy) < self.rows):
                x += dx
                y += dy
                # If any square is blocked in the directional delta
                # then break out of this one and explore the next directional delta
                if self.box_blocked(x, y):
                    break
                moves.append((x, y))

        return moves

    def make_move(self, x, y):
        """Make a move to (x, y) for the active player.
        Block the position of the box on the move.
        Switch to the next player when done.
        Args:
            x (int): X box coordinate.
            y (int): Y box coordinate.
        """
        # assert the box we are moving to is not blocked
        assert not self.box_blocked(x, y)

        # Make the move to the new position and block it
        self.__players_position[self.__active_player] = (x, y)
        self.__block_box(x, y, self.__active_player)

        # Switch the player
        if self.__active_player == Board.PLAYER1:
            self.__active_player = Board.PLAYER2
            self.__inactive_player = Board.PLAYER1
        else:
            self.__active_player = Board.PLAYER1
            self.__inactive_player = Board.PLAYER2

    def make_move_copy(self, x, y):
        """Make a move to (x, y) for the active player returned as a copied game board.
        Block the position of the box on the move.
        Switch to the next player when done.
        Args:
            x (int): X box coordinate.
            y (int): Y box coordinate.
        Returns:
            board_copy (Board): Board object with new state applied.
        """
        board_copy = pickle.loads(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

        # assert the box we are moving to is not blocked
        assert not board_copy.box_blocked(x, y)

        # Make the move to the new position and block it
        board_copy.__players_position[board_copy.__active_player] = (x, y)
        board_copy.__block_box(x, y, board_copy.__active_player)

        # Switch the player
        if board_copy.__active_player == Board.PLAYER1:
            board_copy.__active_player = Board.PLAYER2
            board_copy.__inactive_player = Board.PLAYER1
        else:
            board_copy.__active_player = Board.PLAYER1
            board_copy.__inactive_player = Board.PLAYER2

        return board_copy

    def is_game_over(self):
        """Determine if the game is over.
        Returns:
            The winning player or False.
        """
        # Always check the active player first!
        if not self.get_legal_moves(self.__active_player):
            return self.__inactive_player
        if not self.get_legal_moves(self.__inactive_player):
            return self.__active_player
        return False


class AI(object):
    """Class for AI player.
    Attributes:
        MAX_SCORE (int): Absolute maximum winning / losing score.
    """

    MAX_SCORE = 10000

    @staticmethod
    def score_func(board, winner, player):
        """Score a move for the active player.
        Difference of valid moves left from active player and inactive player.
        Score higher if active player has more moves left than the opponent has.
        Args:
            board (board.Board): Game board object.
            winner (boolean/int): False if game is in play, and int for the winning player.
            player (int): "Player" check as winner.
        Returns:
            score (int): Score for the active player.
        """
        # Terminal heuristic / game over scenario
        if winner:
            if winner == player:
                # "player" won
                return AI.MAX_SCORE
            # "player" lost
            return -AI.MAX_SCORE

        # Non-terminal scoring heuristic
        a_moves = len(board.get_legal_moves(board.active_player))
        o_moves = len(board.get_legal_moves(board.inactive_player))
        return a_moves - o_moves

    @staticmethod
    def negamax(board, depth, player, alpha, beta, score_func):
        """Perform negamax from the perspective of "player" as the active player.
        Sign +1 if "player" is the active player, and sign -1 for opponent.
        Args:
            board (board.Board): Game board object.
            depth (int): The maximum search depth for each state move.
            player (int): "Player" to maximise / check as winner.
            score_func (function pointer): Scoring heuristic.
        Returns:
            best_score (int), best_move (int, int): Best score and associated move for "player".
        """
        player_sign = +1 if board.active_player == player else -1

        winner = board.is_game_over()
        # Game is over or depth is 0, score the move
        if winner or depth == 0:
            return player_sign * score_func(board, winner, player), None

        best_move = None
        best_score = float("-inf")

        # Explore all possible states
        for move in board.get_legal_moves():
            new_board = board.make_move_copy(*move)

            rec_score, current_move = AI.negamax(new_board, depth - 1, player, alpha, beta, score_func)
            current_score = -rec_score

            if current_score > best_score:
                best_score = current_score
                best_move = move

        return best_score, best_move

    @staticmethod
    def abnegamax(board, depth, player, alpha, beta, score_func):
        """Perform abnegamax from the perspective of "player" as the active player.
        This from the Wikipedia site: https://en.wikipedia.org/wiki/Negamax
        Sign +1 if "player" is the active player, and sign -1 for opponent.
        Args:
            board (board.Board): Game board object.
            depth (int): The maximum search depth for each state move.
            player (int): "Player" to maximise / check as winner.
            alpha (int): Lower bound.
            beta (int): Upper bound.
            score_func (function pointer): Scoring heuristic.
        Returns:
            best_score (int), best_move (int, int): Best score and associated move for "player".
        """
        player_sign = +1 if board.active_player == player else -1

        winner = board.is_game_over()
        # Game is over or depth is 0, score the move
        if winner or depth == 0:
            return player_sign * score_func(board, winner, player), None

        best_move = None
        best_score = float("-inf")

        # Explore all possible states
        for move in board.get_legal_moves():
            new_board = board.make_move_copy(*move)

            rec_score, current_move = AI.abnegamax(new_board, depth - 1, player, -beta, -alpha, score_func)
            current_score = -rec_score

            if current_score > best_score:
                best_score = current_score
                best_move = move

            alpha = max(alpha, current_score)
            if alpha >= beta:
                break

        return best_score, best_move

    @staticmethod
    def abnegascout(board, depth, player, alpha, beta, score_func):
        """Perform abnegascout from the perspective of "player" as the active player.
        Sign +1 if "player" is the active player, and sign -1 for opponent.
        Args:
            board (board.Board): Game board object.
            depth (int): The maximum search depth for each state move.
            player (int): "Player" to maximise / check as winner.
            alpha (int): Lower bound.
            beta (int): Upper bound.
            score_func (function pointer): Scoring heuristic.
        Returns:
            best_score (int), best_move (int, int): Best score and associated move for "player".
        """
        player_sign = +1 if board.active_player == player else -1

        winner = board.is_game_over()
        # Game is over or depth is 0, score the move
        if winner or depth == 0:
            return player_sign * score_func(board, winner, player), None

        best_move = None
        best_score = float("-inf")

        b = beta

        # Explore all possible states
        for i, move in enumerate(board.get_legal_moves()):
            new_board = board.make_move_copy(*move)

            rec_score, current_move = AI.abnegascout(new_board, depth - 1, player, -b, -alpha, score_func)
            if rec_score > alpha and rec_score < beta and i > 0:
                rec_score, current_move = AI.abnegascout(new_board, depth - 1, player, -beta, -alpha, score_func)

            current_score = -rec_score

            if current_score > best_score:
                best_score = current_score
                best_move = move

            alpha = max(alpha, current_score)
            if alpha >= beta:
                break
            b = alpha + 1

        return best_score, best_move


def main():
    start_x = 10
    start_y = 10

    box_width = 50
    margin_width = 5

    rows = 5
    columns = 5

    algorithm = 3  # 1 - NegaMax, 2 - NegaMax with alpha-beta pruning, 3 - NegaScout

    func = None
    if algorithm == 1:
        func = AI.negamax
    elif algorithm == 2:
        func = AI.abnegamax
    elif algorithm == 3:
        func = AI.abnegascout

    game = Board(rows, columns)

    # Board/box state to rendering mappings
    box_mapping = {game.PLAYER1: LBLUE,
                   game.PLAYER2: LRED,
                   game.BOX_CLEAR: GREY,
                   game.BOX_BLOCK: BROWN,
                   game.BOX_BLOCKED_MASK: BLACK}  # Masked state is not actually used to draw

    # Player 1 = blue
    player1_colour = BLUE
    # Player 2 = red
    player2_colour = RED

    pygame.init()
    display = pygame.display.set_mode((800, 600), 0, 32)
    pygame.display.set_caption("Isolation")
    visual_board = RenderBoard(display, start_x, start_y, rows, columns, box_mapping, box_width, margin_width)

    render_update = True
    game_over = False
    # AI vs. AI, else player1 = first to move = human
    human_playing = True

    # Record play:
    # 1st element is a tuple with board dimensions and if AI or human player (rows, columns, human_playing)
    # Followed by tuple (active_player, score, move)
    # tuple (active_player, None, move) if human is playing
    record_play = [(rows, columns, human_playing)]

    # Default search depth
    depth = 5

    while True:
        # Only refresh the screen if an action caused a state change
        if render_update:
            # Unset the box blocked bit mask pattern if it is set to get players colour etc.
            board_list = [i & ~game.BOX_BLOCKED_MASK if i != game.BOX_BLOCKED_MASK else
                          game.BOX_BLOCKED_MASK for i in game.board_list]

            display.fill(BLACK)

            # Highlight active player by drawing a margin filled box
            player_pos = game.player1_pos
            if game.active_player == game.PLAYER2:
                player_pos = game.player2_pos
            if player_pos:
                visual_board.draw_margin_box(*player_pos, GREEN)

            # Render the entire board from the board_list
            visual_board.draw_board(board_list)

            # Draw current position of players a darker colour
            player_pos = game.player1_pos
            if player_pos:
                visual_board.draw_box(*player_pos, player1_colour)
            player_pos = game.player2_pos
            if player_pos:
                visual_board.draw_box(*player_pos, player2_colour)

            winner = game.is_game_over()
            if winner:
                print("Player", winner, "wins!")
                game_over = True

            pygame.display.update()
            render_update = False

        # Human goes first, unless AI vs. AI is in play
        if game.active_player == game.PLAYER1 and not human_playing and not game_over:
            score_func = AI.score_func
            start = time.time()
            best_score, best_move = func(game, depth, game.PLAYER1,
                                         float("-inf"), float("inf"), score_func)
            end = time.time()
            print("Best move player 1:", best_move, "score:", best_score, "time:", end - start)

            # Record move
            record_play.append((game.active_player, best_score, best_move))
            game.make_move(*best_move)
            render_update = True

        elif game.active_player == game.PLAYER2 and not game_over:
            score_func = AI.score_func
            start = time.time()
            best_score, best_move = func(game, depth, game.PLAYER2,
                                         float("-inf"), float("inf"), score_func)
            end = time.time()
            print("Best move player 2:", best_move, "score:", best_score, "time:", end - start)

            # Record move
            record_play.append((game.active_player, best_score, best_move))
            game.make_move(*best_move)
            render_update = True

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:  # Key "r" for refresh is pressed
                if event.key == K_r:
                    game = Board(rows, columns)
                    render_update = True
                    game_over = False
                elif event.key == K_q:
                    pygame.quit()
                    sys.exit()
            # Only process a mouse button event if the game is still running
            elif event.type == MOUSEBUTTONDOWN and game.active_player == game.PLAYER1 \
                    and human_playing and not game_over:
                mouse_x, mouse_y = pygame.mouse.get_pos()

                if visual_board.mouse_in_board(mouse_x, mouse_y):
                    box_x, box_y = visual_board.get_box_coord(mouse_x, mouse_y)
                    if not visual_board.mouse_in_margin(mouse_x, mouse_y, box_x, box_y):
                        # Make sure the box selected is not blocked
                        if not game.box_blocked(box_x, box_y) and (box_x, box_y) in game.get_legal_moves():
                            # Record move, set best_score = None for human player
                            record_play.append((game.active_player, None, (box_x, box_y)))
                            # Update the game board state and flip the player
                            game.make_move(box_x, box_y)
                            # Force a render/board refresh
                            render_update = True


main()
