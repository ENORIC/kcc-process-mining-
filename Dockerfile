# Deploys ONLY the dashboard (Layer 3) as a public web app. The heavier
# steps -- fetching real data, hand-labeling the gold set, running Ollama,
# training the classifier -- happen on your own machine, per the README.
# This container just serves the pre-computed outputs + trained model that
# are already checked into the repo.
FROM python:3.11-slim

# graphviz provides the `dot` binary pm4py shells out to when the dashboard
# regenerates a process map live for a crop filter -- Render's native Python
# runtime has no apt/root access to install this, which is the whole reason
# this is a Docker deploy instead of an auto-detected Python one.
RUN apt-get update && apt-get install -y --no-install-recommends graphviz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

COPY . .

ENV PORT=8050
EXPOSE 8050

CMD ["sh", "-c", "cd dashboard && gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:server"]
