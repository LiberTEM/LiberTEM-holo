[Misc] Publish releases on Zenodo
=================================

 * Release and release candidate tags now upload a draft to Zenodo. The record is found by its
   concept ID, as in LiberTEM (secret :code:`LT_RELEASE_UPLOAD_ZENODO_CONCEPT` replaces
   :code:`LT_RELEASE_UPLOAD_ZENODO_PARENT`), and a failed upload fails the CI release job.
 * New script :code:`scripts/zenodo_add_funding` adds the funding entries from
   :code:`packaging/zenodo-funding.json` (AIDAS) to the draft, which Zenodo's deposit API
   can't set.
 * Updated the Zenodo metadata: license GPL-3.0-or-later, grants, keywords, creators and
   contributors. The package keywords in :code:`pyproject.toml` now match the Zenodo keywords.
 * Pinned :code:`bibtexparser<2` for the release scripts; they use the 1.x API.
