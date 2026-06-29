module.exports = {
  apps: [
    {
      name: "moodtune-backend",
      script: "app.py",
      interpreter: process.env.MOODTUNE_PYTHON || "C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python314\\python.exe",
      cwd: "C:\\moodtune\\backend",
      env: {
        MOODTUNE_FRONTEND: "https://anhtaictv.me",
        MOODTUNE_SECRET_KEY: process.env.MOODTUNE_SECRET_KEY,
        MOODTUNE_ADMIN_KEY: process.env.MOODTUNE_ADMIN_KEY,
      },
      watch: false,
      autorestart: true,
    },
  ],
};
