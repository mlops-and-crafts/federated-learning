from flask import Flask

import cfg

app = Flask(__name__)


@app.route("/")
def index():
    return "<h1>Federated Dashboard</h1>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=cfg.DASHBOARD_PORT)
