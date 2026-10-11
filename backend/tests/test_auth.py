def test_register_user(client):
    res = client.post("/api/auth/register", json={"email": "a@example.com", "password": "pass1234"})
    assert res.status_code == 201
    assert res.json()["email"] == "a@example.com"


def test_register_duplicate_user_fails(client):
    client.post("/api/auth/register", json={"email": "dup@example.com", "password": "pass1234"})
    res = client.post("/api/auth/register", json={"email": "dup@example.com", "password": "pass1234"})
    assert res.status_code == 409


def test_login_success(client):
    client.post("/api/auth/register", json={"email": "b@example.com", "password": "pass1234"})
    res = client.post("/api/auth/login", json={"email": "b@example.com", "password": "pass1234"})
    assert res.status_code == 200
    assert "access_token" in res.json()


def test_login_wrong_password_fails(client):
    client.post("/api/auth/register", json={"email": "c@example.com", "password": "pass1234"})
    res = client.post("/api/auth/login", json={"email": "c@example.com", "password": "wrongpass"})
    assert res.status_code == 401


def test_protected_route_requires_token(client):
    res = client.get("/api/datasets")
    assert res.status_code == 401
