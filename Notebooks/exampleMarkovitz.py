from Models.Allocation import OptimalAllocation

w = OptimalAllocation(["AAPL","MSFT","GOOG","AMZN"], method="Markovitz", allow_short=False)
print("Optimal portfolio weights:")
for ticker, weight in zip(["AAPL", "MSFT", "GOOG", "AMZN"], w):
    print(f"  {ticker}: {weight*100:.4f}"+"%")
print(f"Total weight allocated: {w.sum():.4f}")

w_french = OptimalAllocation(
    ["ENGI.PA", "TTE.PA", "SAN.PA", "ACA.PA"],
    method="Markovitz",
    allow_short=False
)

print("Optimal portfolio weights:")
for ticker, weight in zip(["ENGI.PA", "TTE.PA", "SAN.PA", "ACA.PA"], w_french):
    print(f"  {ticker}: {weight*100:.4f}"+"%")
print(f"Total weight allocated: {w_french.sum():.4f}")


w_french_withReturn = OptimalAllocation(
    ["ENGI.PA", "TTE.PA", "SAN.PA", "ACA.PA"],
    method="Markovitz",
    allow_short=False,
    target_return=0.10  # 10% annualized target return
)

print("Optimal portfolio weight with target annual return of 10%:")
for ticker, weight in zip(["ENGI.PA", "TTE.PA", "SAN.PA", "ACA.PA"],w_french_withReturn):
    print(f"  {ticker}: {weight*100:.4f}"+"%")
print(f"Total weight allocated: {w_french_withReturn.sum():.4f}")

