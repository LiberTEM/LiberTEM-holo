[Bugfix] No "not a git repository" message on import
=====================================================

 * Importing an installed ``libertem_holo`` no longer prints
   ``fatal: not a git repository``. Wheels and sdists now include the git
   revision they were built from (``libertem_holo.revision``), and outside of a
   source checkout, git is no longer called (:issue:`99`).
