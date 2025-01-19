# During setup of games, this should be set to the index of the game that is currently being set up.
# Any Pubs and Subs being created as part of that game will then know the index of the game they belong to (as it's set
# in their constructors, see below)
# index -1 refers to the overall Runtime object containing the games
# indices >= 0 refer to the Game objects
current_game_index: int = -1
