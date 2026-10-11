import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Database,
  Boxes,
  SlidersHorizontal,
  Activity,
  FlaskConical,
  MessageSquare,
  LogOut,
} from "lucide-react";
import { useAuth } from "../hooks/useAuth";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/datasets", label: "Datasets", icon: Database },
  { to: "/models", label: "Models", icon: Boxes },
  { to: "/fine-tuning", label: "Fine-Tuning", icon: SlidersHorizontal },
  { to: "/training-jobs", label: "Training Jobs", icon: Activity },
  { to: "/experiments", label: "Experiments", icon: FlaskConical },
  { to: "/playground", label: "Playground", icon: MessageSquare },
];

export function Sidebar() {
  const { logout } = useAuth();

  return (
    <aside className="w-60 shrink-0 h-screen sticky top-0 border-r border-base-800 bg-base-900 flex flex-col">
      <div className="px-5 py-5 border-b border-base-800">
        <div className="flex items-center gap-2">
          <div className="h-7 w-7 rounded-md bg-accent flex items-center justify-center text-xs font-bold">
            FT
          </div>
          <div>
            <p className="text-sm font-semibold leading-tight">FineTune Studio</p>
            <p className="text-[11px] text-slate-500 leading-tight">local-first LLM ops</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                isActive
                  ? "bg-accent-soft text-accent font-medium"
                  : "text-slate-400 hover:text-slate-100 hover:bg-base-800"
              }`
            }
          >
            <Icon size={16} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="px-3 py-4 border-t border-base-800">
        <button
          onClick={logout}
          className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-slate-400 hover:text-bad hover:bg-base-800 w-full transition-colors"
        >
          <LogOut size={16} />
          Sign out
        </button>
      </div>
    </aside>
  );
}
