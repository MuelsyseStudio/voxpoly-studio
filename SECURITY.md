# Security & Responsible AI Policy

## Responsible Voice Cloning Guidelines

VoxPoly Studio is built for legitimate multilingual content creation, accessibility, education, podcasting, and localization. By using this software, you agree to:

1. **Consent First**: Only clone your own voice or voices for which you have obtained explicit, informed consent from the speaker.
2. **No Impersonation or Fraud**: Never use generated audio or dubbed videos for impersonation, social engineering, defamation, political disinformation, or bypassing voice biometrics.
3. **Transparency & Watermarking**: Keep the built-in imperceptible acoustic watermark enabled (`enable_watermark: true`) whenever publishing or sharing generated media, and disclose synthetic audio where required by applicable laws.

## Deployment Security

- **Local First**: By default, VoxPoly Studio binds strictly to `127.0.0.1:8765` so it is not exposed on your local network.
- **Public Tunnel Protection**: When exposing the studio via Cloudflare Tunnel or reverse proxy (e.g., on Google Colab), always set `VOXPOLY_ACCESS_TOKEN` to require passcode authentication on all `/api/*` endpoints.
- **Secrets**: Never commit `data/settings.local.json` or `.env` files containing third-party LLM API keys. They are ignored by `.gitignore`.

## Reporting Vulnerabilities

If you discover a security issue in VoxPoly Studio, please open a private security advisory on GitHub instead of posting exploit details publicly.
