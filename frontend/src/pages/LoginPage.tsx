import { FormEvent, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await login(email, password);
      navigate("/");
    } catch (err: any) {
      setError(err?.response?.data?.error?.message ?? "Login failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-base-950 px-4">
      <div className="w-full max-w-sm">
        <div className="mb-8 text-center">
          <div className="h-10 w-10 rounded-lg bg-accent mx-auto mb-3 flex items-center justify-center font-bold">
            FT
          </div>
          <h1 className="text-lg font-semibold">Sign in to FineTune Studio</h1>
          <p className="text-sm text-slate-500 mt-1">Local-first LLM fine-tuning &amp; evaluation</p>
        </div>
        <form onSubmit={handleSubmit} className="card space-y-4">
          {error && <p className="text-sm text-bad">{error}</p>}
          <div>
            <label className="label">Email</label>
            <input className="input" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
          </div>
          <div>
            <label className="label">Password</label>
            <input
              className="input"
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          <button className="btn-primary w-full" type="submit" disabled={loading}>
            {loading ? "Signing in…" : "Sign in"}
          </button>
        </form>
        <p className="text-sm text-slate-500 text-center mt-4">
          No account? <Link to="/register" className="text-accent">Register</Link>
        </p>
      </div>
    </div>
  );
}
