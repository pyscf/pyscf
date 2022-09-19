#!/bin/bash

set -e

# Get latest version tag
get_last_tag() {
  curl --silent "https://api.github.com/repos/pyscf/pyscf/releases/latest" | sed -n 's/.*"tag_name": "v\(.*\)",.*/\1/p'
}
last_version=$(get_last_tag)
#last_version=$(git tag | tail -1 | sed 's/v//')
echo Last version: $last_version

# Get current version tag
cur_version=$(sed -n "/^__version__ =/s/.*'\(.*\)'/\1/p" pyscf/__init__.py)
if [ -z "$cur_version" ]; then
  cur_version=$(sed -n '/^__version__ =/s.*"\(.*\)"/\1/p' pyscf/__init__.py)
fi
echo Current version: $cur_version

# Create version tag
if [[ -n "$last_version" ]] && [[ -n "$cur_version" ]] && [[ "$cur_version" != "$last_version" ]]; then
  git config user.name "Github Actions"
  git config user.email "github-actions@users.noreply.github.com"
  version_tag=v"$cur_version"

  # Extract release info from CHANGELOG
  #release_info=$(sed -n "/$cur_version/,/$last_version/p" CHANGELOG | head -n -2)
  sed -n "/^PySCF $cur_version/,/^PySCF $last_version/p" CHANGELOG | tail -n +3 | sed -e '/^PySCF /,$d' | head -n -2 > RELEASE.md
  echo "::set-output name=version_tag::$version_tag"
fi
