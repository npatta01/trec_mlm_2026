document.querySelectorAll("[data-hub-filter-group]").forEach((group) => {
  const rows = [...group.querySelectorAll("[data-hub-filter-row]")];
  const search = group.querySelector("[data-hub-filter]");
  const status = group.querySelector("[data-hub-status]");
  const scope = group.querySelector("[data-hub-scope]");
  const live = group.querySelector("[data-hub-filter-live]");

  const apply = () => {
    const needle = (search?.value || "").trim().toLowerCase();
    let visible = 0;
    rows.forEach((row) => {
      const matchesSearch = !needle || row.dataset.search.includes(needle);
      const matchesStatus = !status?.value || row.dataset.status === status.value;
      const matchesScope = !scope?.value || row.dataset.scope === scope.value;
      row.hidden = !(matchesSearch && matchesStatus && matchesScope);
      if (!row.hidden) visible += 1;
    });
    if (live) live.textContent = `${visible} results`;
  };

  [search, status, scope].filter(Boolean).forEach((control) => {
    control.addEventListener("input", apply);
    control.addEventListener("change", apply);
  });
  apply();
});
