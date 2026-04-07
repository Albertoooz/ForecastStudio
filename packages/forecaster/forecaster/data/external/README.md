# External Data Directory

Place CSV or Parquet files here for automatic discovery by ExternalDataAgent.

## Expected tables:

### holidays_pl.csv
Columns: date, name, type
```
date,name,type
2024-01-01,Nowy Rok,national
2024-01-06,Trzech Króli,national
...
```

### macro_eu.csv
Columns: date, gdp_growth, inflation, unemployment
```
date,gdp_growth,inflation,unemployment
2024-01-01,0.3,2.1,6.2
2024-02-01,0.4,2.0,6.1
...
```

Files are auto-detected by the ExternalDataAgent and suggested as joins
based on date overlap with your primary data.
