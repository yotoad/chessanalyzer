import chess
import chess.svg
import chess.pgn
import chess.engine
from tkinter import *
from tkinter import simpledialog
from PIL import Image, ImageTk
import io
import threading
import platform
import os

class ChessMoveAnalyzer:
    def __init__(self, moves, root=None):
        self.moves = moves
        self.board = chess.Board()
        
        # Platform-specific Stockfish path
        system = platform.system().lower()
        if system == "windows":
            stockfish_path = "./stockfish/stockfish-windows-x86-64-avx2.exe"
        elif system == "linux":
            stockfish_path = "./stockfish/stockfish-ubuntu-x86-64-avx2"
        elif system == "darwin":
            stockfish_path = "./stockfish/stockfish-macos"
        else:
            raise RuntimeError("Unsupported operating system")

        if not os.path.exists(stockfish_path):
            raise FileNotFoundError(f"Stockfish not found at {stockfish_path}")
            
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.analysis = []
        self.current_move_index = 0
        self.root = root

    def analyze_game(self, progress_callback=None):
        """Analyze each move in the game"""
        for i, move_san in enumerate(self.moves):
            try:
                move = self.board.parse_san(move_san)
            except chess.IllegalMoveError:
                print(f"Illegal move: {move_san}")
                break
                
            # Analyze position before the move
            info = self.engine.analyse(self.board, chess.engine.Limit(time=0.1))
            prev_score = info["score"].relative.score(mate_score=10000)
            
            self.board.push(move)
            
            # Analyze position after the move
            info = self.engine.analyse(self.board, chess.engine.Limit(time=0.1))
            new_score = info["score"].relative.score(mate_score=10000)
            
            # Determine move quality
            score_diff = new_score - prev_score
            best_move = info.get("pv", [move])[0]
            
            move_data = {
                "move": move_san,
                "score_before": prev_score,
                "score_after": new_score,
                "score_diff": score_diff,
                "best_move": self.board.san(best_move),
                "position_fen": self.board.fen()
            }
            
            self.analysis.append(move_data)
            if progress_callback:
                progress_callback(i/len(self.moves))
    
    def get_move_quality(self, move_index):
        """Classify move quality"""
        if move_index >= len(self.analysis):
            return None
            
        data = self.analysis[move_index]
        score_diff = data["score_diff"]
        
        if score_diff is None:
            return "unknown"
        
        if score_diff > 200:
            return "excellent"
        elif score_diff > 50:
            return "good"
        elif score_diff < -200:
            return "blunder"
        elif score_diff < -50:
            return "bad"
        else:
            return "neutral"
    
    def get_move_evaluation(self, move_index):
        """Get evaluation text for a move"""
        if move_index >= len(self.analysis):
            return ""
            
        data = self.analysis[move_index]
        quality = self.get_move_quality(move_index)
        
        if quality == "excellent":
            return f"Excellent move! ({data['move']})"
        else:
            best_move = data["best_move"]
            centipawn_loss = max(0, -data["score_diff"])
            
            quality_text = {
                "good": "Good move",
                "neutral": "Reasonable move",
                "bad": "Bad move",
                "blunder": "Blunder"
            }.get(quality, "Move")
            
            return (f"{quality_text} ({data['move']})\n"
                    f"Best was {best_move} (losing {centipawn_loss/100:.1f} pawns)\n"
                    f"Evaluation: {self.score_to_text(data['score_after'])}")
    
    def score_to_text(self, score):
        """Convert centipawn score to human-readable text"""
        if score is None:
            return "Unknown"
        
        abs_score = abs(score)
        if abs_score > 500:
            return "Decisive advantage"
        elif abs_score > 200:
            return "Significant advantage"
        elif abs_score > 100:
            return "Moderate advantage"
        elif abs_score > 50:
            return "Slight advantage"
        else:
            return "Even position"
    
    def close(self):
        self.engine.quit()

class ChessGUI:
    def __init__(self, root, move_analyzer):
        self.root = root
        self.analyzer = move_analyzer
        self.current_move = 0
        self.after_id = None
        self.modified_moves = set()
        
        # Set up GUI
        self.setup_gui()
        
        # Start analysis in background
        self.start_analysis()
        
        # Display initial position
        self.update_display()

    def setup_gui(self):
        """Initialize all GUI components"""
        self.root.title("Chess Move Analyzer")
        
        # Board display
        self.board_label = Label(self.root)
        self.board_label.pack()
        
        # Evaluation text display
        self.eval_text = Text(self.root, height=6, width=50, wrap=WORD)
        self.eval_text.pack(pady=10)
        
        # Style selection
        self.style_frame = Frame(self.root)
        self.style_frame.pack(pady=10)
        
        # Color scheme selection
        Label(self.style_frame, text="Colors:").pack(side=LEFT)
        self.color_schemes = {
            "Classic": ("#F0D9B5", "#B58863"),
            "Blue": ("#DEE3E6", "#8CA2AD"),
            "Green": ("#FFFFDD", "#86A666"),
            "Gray": ("#EAEAEA", "#8E8E8E")
        }
        self.color_var = StringVar(value="Classic")
        OptionMenu(self.style_frame, self.color_var, *self.color_schemes.keys(),
                 command=self.update_display).pack(side=LEFT, padx=5)

        # Move navigation
        self.nav_frame = Frame(self.root)
        self.nav_frame.pack(pady=10)
        
        self.prev_button = Button(self.nav_frame, text="← Previous", command=self.prev_move)
        self.prev_button.pack(side=LEFT, padx=5)
        
        self.next_button = Button(self.nav_frame, text="Next →", command=self.next_move)
        self.next_button.pack(side=LEFT, padx=5)
        
        # Move input controls
        self.move_frame = Frame(self.root)
        self.move_frame.pack(pady=10)
        
        Label(self.move_frame, text="Move").pack(side=LEFT)
        self.move_number_label = Label(self.move_frame, text="1")
        self.move_number_label.pack(side=LEFT)
        
        self.move_var = StringVar()
        self.move_entry = Entry(self.move_frame, textvariable=self.move_var, width=10)
        self.move_entry.pack(side=LEFT, padx=5)
        self.move_entry.bind("<Return>", lambda e: self.play_custom_move())
        
        self.play_button = Button(self.move_frame, text="Play Move (Enter)", 
                                command=self.play_custom_move)
        self.play_button.pack(side=LEFT)
        
        # Save/Load buttons
        self.save_load_frame = Frame(self.root)
        self.save_load_frame.pack(pady=10)
        
        self.save_button = Button(self.save_load_frame, text="Save Game", 
                                command=self.save_game)
        self.save_button.pack(side=LEFT, padx=5)
        
        self.load_button = Button(self.save_load_frame, text="Load Game", 
                                command=self.load_game)
        self.load_button.pack(side=LEFT, padx=5)
        
        # Status bar
        self.status_var = StringVar()
        self.status_bar = Label(self.root, textvariable=self.status_var, relief=SUNKEN, anchor=W)
        self.status_bar.pack(fill=X, side=BOTTOM)
        
        # Key bindings
        self.root.bind("<Left>", lambda e: self.prev_move())
        self.root.bind("<Right>", lambda e: self.next_move())
        self.root.bind("<Up>", lambda e: self.move_entry.focus())

    def safe_update_status(self, text):
        """Thread-safe way to update status"""
        if self.root.winfo_exists():
            self.root.after(0, lambda: self.status_var.set(text))

    def start_analysis(self):
        """Run analysis in background thread"""
        def analysis_task():
            try:
                self.safe_update_status("Analyzing moves...")
                self.analyzer.analyze_game(
                    progress_callback=lambda p: self.safe_update_status(f"Analyzing... {p*100:.0f}%")
                )
                self.safe_update_status("Analysis complete")
            except Exception as e:
                self.safe_update_status(f"Error: {str(e)}")
            finally:
                self.update_display()

        analysis_thread = threading.Thread(target=analysis_task, daemon=True)
        analysis_thread.start()

    def generate_svg(self, board):
        """Generate SVG with current style settings"""
        colors = self.color_schemes[self.color_var.get()]
        return chess.svg.board(
            board=board,
            arrows=self.get_arrows(),
            size=400,
            colors={
                "square light": colors[0],
                "square dark": colors[1]
            }
        )
    
    def get_arrows(self):
        """Get arrows for current position"""
        arrows = []
        if 0 < self.current_move <= len(self.analyzer.analysis):
            if self.current_move-1 in self.modified_moves:
                # Show original best move in red
                move_data = self.analyzer.analysis[self.current_move-1]
                best_move_uci = chess.Board(move_data["position_fen"]).parse_san(
                    move_data["best_move"]).uci()
                arrows.append(chess.svg.Arrow(
                    tail=chess.parse_square(best_move_uci[:2]),  # Changed from_ to tail
                    head=chess.parse_square(best_move_uci[2:]),   # Changed to_ to head
                    color="#ff000088"
                ))
            elif self.analyzer.get_move_quality(self.current_move-1) != "excellent":
                # Show suggested move in green
                move_data = self.analyzer.analysis[self.current_move-1]
                best_move_uci = chess.Board(move_data["position_fen"]).parse_san(
                    move_data["best_move"]).uci()
                arrows.append(chess.svg.Arrow(
                    tail=chess.parse_square(best_move_uci[:2]),  # Changed from_ to tail
                    head=chess.parse_square(best_move_uci[2:]),   # Changed to_ to head
                    color="#00ff0088"
                ))
        return arrows

    def update_display(self, *args):
        """Schedule display updates in main thread"""
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(100, self._update_display)

    def _update_display(self):
        """Actual display update (runs in main thread)"""
        try:
            # Update move number and color
            move_number = (self.current_move // 2) + 1
            move_color = "White" if self.current_move % 2 == 0 else "Black"
            self.move_number_label.config(text=f"{move_number}. {'White' if move_color == 'White' else 'Black'}")

            # Reconstruct board state up to current move
            board = chess.Board()
            for i in range(self.current_move):
                if i < len(self.analyzer.moves):
                    try:
                        move = board.parse_san(self.analyzer.moves[i])
                        board.push(move)
                    except chess.IllegalMoveError:
                        print(f"Illegal move at position {i}: {self.analyzer.moves[i]}")
                        continue

            # Display board
            svg_data = chess.svg.board(board=board, size=400)
            try:
                png_data = self.svg_to_png(svg_data)
                self.photo = ImageTk.PhotoImage(png_data)
                self.board_label.config(image=self.photo)
            except Exception as e:
                print(f"Rendering error: {e}")
                self.board_label.config(text="Board display unavailable")

            # Update move entry with next move if available
            if self.current_move < len(self.analyzer.moves):
                self.move_var.set(self.analyzer.moves[self.current_move])
            else:
                self.move_var.set("")

            # Update evaluation text if analysis exists
            if 0 < self.current_move <= len(self.analyzer.analysis):
                move_data = self.analyzer.analysis[self.current_move-1]
                quality = self.analyzer.get_move_quality(self.current_move-1)
                eval_text = self.analyzer.get_move_evaluation(self.current_move-1)
                
                self.eval_text.delete(1.0, END)
                self.configure_text_tags()
                self.eval_text.insert(END, eval_text)
                first_line_end = eval_text.find('\n') if '\n' in eval_text else len(eval_text)
                self.eval_text.tag_add(quality, "1.0", f"1.{first_line_end}")
            else:
                self.eval_text.delete(1.0, END)
                self.eval_text.insert(END, "Initial position" if self.current_move == 0 else "Move played, analyzing...")

            # Update status and buttons
            self.status_var.set(f"Move {self.current_move + 1}/{len(self.analyzer.moves) + 1}")
            self.prev_button.config(state=NORMAL if self.current_move > 0 else DISABLED)
            self.next_button.config(state=NORMAL if self.current_move < len(self.analyzer.moves) else DISABLED)

        except Exception as e:
            print(f"Display update error: {e}")
        finally:
            self.after_id = None
    
    def update_evaluation_text(self):
        """Update the evaluation text widget"""
        self.eval_text.delete(1.0, END)
        
        if self.current_move == 0:
            self.eval_text.insert(END, "Initial position")
            return
            
        quality = self.analyzer.get_move_quality(self.current_move-1)
        eval_text = self.analyzer.get_move_evaluation(self.current_move-1)
        
        # Configure text colors
        self.configure_text_tags()
        
        # Insert text with coloring
        self.eval_text.insert(END, eval_text)
        first_line_end = eval_text.find('\n') if '\n' in eval_text else len(eval_text)
        self.eval_text.tag_add(quality, "1.0", f"1.{first_line_end}")

    def configure_text_tags(self):
        """Set up text coloring tags"""
        colors = {
            "excellent": "green",
            "good": "blue",
            "neutral": "black",
            "bad": "orange",
            "blunder": "red"
        }
        for tag, color in colors.items():
            self.eval_text.tag_config(tag, foreground=color)

    def svg_to_png(self, svg_data):
        """Convert SVG to PNG with fallback options"""
        try:
            # Try cairosvg first
            import cairosvg
            png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
            return Image.open(io.BytesIO(png_data))
        except ImportError:
            # Fallback to svglib if cairo not available
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
            drawing = svg2rlg(io.BytesIO(svg_data.encode('utf-8')))
            png_data = renderPM.drawToString(drawing, fmt="PNG")
            return Image.open(io.BytesIO(png_data))

    def play_custom_move(self):
        """Play user-specified move with visual feedback"""
        if self.current_move >= len(self.analyzer.moves):
            return
            
        custom_move = self.move_var.get().strip()
        if not custom_move:
            return
            
        try:
            # Validate move
            board = chess.Board()
            for i in range(self.current_move):
                board.push_san(self.analyzer.moves[i])
            board.parse_san(custom_move)
            
            # Update game state
            if self.analyzer.moves[self.current_move] != custom_move:
                self.modified_moves.add(self.current_move)
            self.analyzer.moves[self.current_move] = custom_move
            self.next_move()
            
        except chess.IllegalMoveError:
            self.status_var.set(f"Invalid move: {custom_move}")
            self.move_entry.config(bg="#FFCCCC")  # Light red for errors
            self.root.after(1000, lambda: self.move_entry.config(bg="white"))

    def next_move(self):
        """Move to next position in game"""
        if self.current_move < len(self.analyzer.moves):
            self.current_move += 1
            self.update_display()

    def prev_move(self):
        """Move to previous position in game"""
        if self.current_move > 0:
            self.current_move -= 1
            self.update_display()
            
    def save_game(self):
        """Save current game to PGN file"""
        game = chess.pgn.Game()
        node = game
        
        for move_san in self.analyzer.moves:
            node = node.add_variation(node.board().parse_san(move_san))
        
        pgn_str = str(game)
        filename = filedialog.asksaveasfilename(
            defaultextension=".pgn",
            filetypes=[("PGN Files", "*.pgn"), ("All Files", "*.*")]
        )
        if filename:
            with open(filename, "w") as f:
                f.write(pgn_str)
            self.status_var.set(f"Game saved to {filename}")

    def load_game(self):
        """Load game from PGN file"""
        filename = filedialog.askopenfilename(
            filetypes=[("PGN Files", "*.pgn"), ("All Files", "*.*")]
        )
        if filename:
            with open(filename) as f:
                game = chess.pgn.read_game(f)
                moves = [self.analyzer.board.san(move.move) for move in game.mainline_moves()]
                self.analyzer = ChessMoveAnalyzer(moves, self.root)
                self.current_move = 0
                self.modified_moves = set()
                self.update_display()
                self.status_var.set(f"Loaded {filename}")

    def on_closing(self):
        """Clean up when closing window"""
        self.analyzer.close()
        self.root.destroy()

def get_moves_from_user():
    """Prompt user to paste moves and clean the input"""
    root = Tk()
    root.withdraw()  # Hide the main window
    
    instructions = """Please paste the chess moves (without quotes):
Example: e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Bc5 O-O d6
"""
    move_str = simpledialog.askstring("Input Moves", instructions, parent=root)
    root.destroy()
    
    if not move_str:
        print("No moves entered. Using sample game.")
        return [
            "e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "d3", "Bc5", 
            "O-O", "d6", "h3", "O-O", "Nc3", "a6", "a4", "h6"
        ]
    
    # Clean the input - remove numbers, dots, extra spaces
    cleaned = ' '.join(move_str.replace('.', ' ').split())
    return [move for move in cleaned.split() if not move.isdigit()]

if __name__ == "__main__":
    # Example game
    sample_moves = [
        "e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "d3", "Bc5", 
        "O-O", "d6", "h3", "O-O", "Nc3", "a6", "a4", "h6", 
        "Be3", "Bxe3", "fxe3", "Be6", "Bxe6", "fxe6", "Qe1", "Qe8", 
        "Qg3", "Kh8", "Nh4", "Nd8", "Rf7", "Qxf7", "Nf5", "exf5", 
        "Qxc7", "Qf6", "exf5", "Qxf5", "Qxd8+", "Kh7", "Qxd6", "Qg5", 
        "Qxe6", "Qg2#"
    ]
    
    moves = get_moves_from_user()
    print("Analyzing moves:", ' '.join(moves))
    
    root = Tk()
    analyzer = ChessMoveAnalyzer(moves, root)
    gui = ChessGUI(root, analyzer)
    root.protocol("WM_DELETE_WINDOW", gui.on_closing)
    root.mainloop()