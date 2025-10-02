File "/mount/src/cryoto_dashboard_full_product.py/crypto_dashboard_full_product.py", line 141, in <module>
    df, latest, alerts = get_technical_signals(ticker)
                         ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
File "/mount/src/cryoto_dashboard_full_product.py/crypto_dashboard_full_product.py", line 40, in get_technical_signals
    df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], close).average_true_range()
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/ta/volatility.py", line 44, in __init__
    self._run()
    ~~~~~~~~~^^
File "/home/adminuser/venv/lib/python3.13/site-packages/ta/volatility.py", line 48, in _run
    true_range = self._true_range(self._high, self._low, close_shift)
File "/home/adminuser/venv/lib/python3.13/site-packages/ta/utils.py", line 45, in _true_range
    true_range = pd.DataFrame(data={"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
                 ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/core/frame.py", line 782, in __init__
    mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/core/internals/construction.py", line 503, in dict_to_mgr
    return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/core/internals/construction.py", line 114, in arrays_to_mgr
    index = _extract_index(arrays)
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/core/internals/construction.py", line 667, in _extract_index
    raise ValueError("If using all scalar values, you must pass an index")
