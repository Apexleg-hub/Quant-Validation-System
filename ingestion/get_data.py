import MetaTrader5 as mt5
import pandas as pd

# Initialize connection to the MT5 terminal
if not mt5.initialize():
    print("MetaTrader5 initialization failed")
    mt5.shutdown()
    quit()
else:
    print("MT5 Initialized successfully")

# ... (data fetching code will go here) ...

# Always shut down the connection when done
mt5.shutdown()

