from werkzeug.security import generate_password_hash
from secrets import token_urlsafe

password = "awdwa"

print(generate_password_hash(password))


print(f"\nRandom tokens for your API keys and secrets: ")
for x in range(3):
    print(token_urlsafe())

