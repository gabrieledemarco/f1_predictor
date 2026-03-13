#!/usr/bin/env python3
"""
Test MongoDB connection using current .env
"""

import os
import sys
sys.path.insert(0, '.')

from core.db import get_db_direct

print("Testing MongoDB connection...")
print(f"MONGO_URI from .env: {os.environ.get('MONGO_URI', 'Not set')}")

# Load .env manually
from dotenv import load_dotenv
load_dotenv()

uri = os.environ.get('MONGO_URI')
if not uri:
    print("ERROR: MONGO_URI not found in .env")
    sys.exit(1)

print(f"URI (hidden): {uri[:30]}...")

db = get_db_direct()
if db is None:
    print("FAILED: MongoDB connection failed")
    sys.exit(1)
else:
    print(f"SUCCESS: Connected to database '{db.name}'")
    # List collections
    cols = db.list_collection_names()
    print(f"Collections ({len(cols)}): {cols[:5]}")
    sys.exit(0)