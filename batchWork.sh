while True; do
  git pull origin main
  python CodingBenchmark.py --parallel
  git add .
  git commit -am "Automated commit"
  git pull origin main
  git reset
  git checkout .
  git clean -f
  git clean -fd
  git push origin main
done