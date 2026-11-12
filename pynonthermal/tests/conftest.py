import os

# Plot tests use files and do not need a desktop session.
# Set the backend before a test imports pyplot.
os.environ["MPLBACKEND"] = "Agg"
