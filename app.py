from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock_symbol = request.form["stock_symbol"]
        stock_data = yf.download(stock_symbol, start="2020-01-01", end="2023-01-01")
        stock_data = stock_data[["Close"]].dropna()

        stock_data["Prediction"] = stock_data["Close"].shift(-1)
        stock_data.dropna(inplace=True)
        X = stock_data[["Close"]]
        y = stock_data["Prediction"]

        X_train, X_test, y_train, y_test = X[:-30], X[-30:], y[:-30], y[-30:]

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        plt.figure(figsize=(10, 5))
        plt.plot(stock_data.index[-30:], y_test, label="Actual Prices", color="blue")
        plt.plot(stock_data.index[-30:], predictions, label="Predicted Prices", linestyle="--", color="red")
        plt.title(f"Stock Price Prediction for {stock_symbol.upper()}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.savefig("static/plot.png")
        plt.close()

        return redirect(url_for("results", stock_symbol=stock_symbol))

    return render_template("index.html")

@app.route("/results")
def results():
    stock_symbol = request.args.get("stock_symbol")
    stock_data = yf.download(stock_symbol, start="2020-01-01", end="2023-01-01")
    stock_data = stock_data[["Close"]].dropna()

    mean_price = stock_data["Close"].mean().item()
    max_price = stock_data["Close"].max().item()
    min_price = stock_data["Close"].min().item()
    last_price = stock_data["Close"].iloc[-1].item()

    previous_price = stock_data["Close"].iloc[-2].item()
    price_change = last_price - previous_price
    change_direction = "increased" if price_change > 0 else "decreased"
    change_percentage = (price_change / previous_price) * 100 if previous_price != 0 else 0

    if change_percentage > 5:
        summary = "The stock shows strong upward momentum, indicating positive investor sentiment and potential for continued growth."
    elif 0 < change_percentage <= 5:
        summary = "The stock has experienced modest growth, suggesting stable performance but limited upward potential."
    elif change_percentage == 0:
        summary = "The stock price has remained stable, indicating a balanced market perception without significant volatility."
    else:
        summary = "The stock has experienced a decline, which may raise concerns about its future performance and market confidence."

    return render_template("results.html",
                           stock_symbol=stock_symbol,
                           mean_price=mean_price,
                           max_price=max_price,
                           min_price=min_price,
                           last_price=last_price,
                           price_change=abs(price_change),
                           change_direction=change_direction,
                           change_percentage=abs(change_percentage),
                           summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
