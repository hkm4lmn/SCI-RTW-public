import React from "react";

// Sidebar icons: simple 20px line drawings, decorative (the label carries
// the meaning), so they are hidden from screen readers.
const PATHS = {
  dashboard: <><rect x="3" y="3" width="6" height="8" rx="1" /><rect x="11" y="3" width="6" height="5" rx="1" /><rect x="3" y="13" width="6" height="4" rx="1" /><rect x="11" y="10" width="6" height="7" rx="1" /></>,
  patients: <><circle cx="7.5" cy="7" r="2.8" /><path d="M2.5 16.5c.5-3 2.4-4.5 5-4.5s4.5 1.500 5 4.5" /><circle cx="14.5" cy="8" r="2" /><path d="M14.200 12.200c1.900.1 3 1.300 3.400 3.300" /></>,
  predict: <><path d="M3 16.500 8 10l3 3 6-8.500" /><path d="M13.500 4.500H17V8" /></>,
  insights: <><path d="M3 17V9" /><path d="M8 17V4" /><path d="M13 17v-6" /><path d="M17.500 17H2.500" /></>,
  history: <><circle cx="10" cy="10" r="7" /><path d="M10 6v4.300l2.800 1.700" /></>,
  users: <><circle cx="10" cy="7" r="3" /><path d="M4 17c.6-3.300 2.900-5 6-5s5.400 1.700 6 5" /></>,
  logout: <><path d="M8 3H4.500A1.500 1.500 0 0 0 3 4.500v11A1.500 1.500 0 0 0 4.500 17H8" /><path d="M12.500 13.500 16 10l-3.500-3.500" /><path d="M16 10H7.500" /></>,
};

export default function Icon({ name }) {
  return (
    <svg className="icon" viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor"
         strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {PATHS[name]}
    </svg>
  );
}
