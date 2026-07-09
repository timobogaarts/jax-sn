'''
This tests all operators using a unified setup based on a 4-group HCPB breeding blanket model. 

This setup is defined in bb_setup.py; conftest.py provides a fixture to load it (saving recompilation of the setup).

The overall strategy is to have single operators test consistency internally, and have composited operator use one particular implementation.
'''