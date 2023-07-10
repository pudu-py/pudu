Contributing
------------
Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Report Bugs
^^^^^^^^^^^

Report bugs at https://github.com/pudu-py/pudu/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
^^^^^^^^

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
^^^^^^^^^^^^^^^^^^

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
^^^^^^^^^^^^^^^^^^^

`pudu` could always use more documentation, whether as part of the
official `pudu` docs or in docstrings.

Submit Feedback
^^^^^^^^^^^^^^^

The best way to send feedback is to file an issue at https://github.com/pudu-py/pudu/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
^^^^^^^^^^^^

Ready to contribute? Here's how to set up `pudu` for local development.

1. Fork the `pudu` repo on GitHub.
2. Clone your fork locally::

        $ git clone git@github.com:your_name_here/pudu.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

        $ mkvirtualenv pudu
        $ cd pudu/
        $ python setup.py develop

4. Create a branch for local development to make changes locally::

        $ git checkout -b name-of-your-bugfix-or-feature

5. When you're done making changes, check that your changes pass flake8 and the tests, including testing other Python versions with tox::

        $ flake8 pudu tests
        $ pytest
        $ tox

6. Run unittest, check the test coverage, and create a coverage report in the `tests` folder::

        $ coverage run -m unittest test_pudu.py
        $ coverage xml

Independently of what you do, all tests must pass. I you add functionality, like `functions`
or `classes`, the they must include a test.

7. Commit your changes and push your branch to GitHub::

        $ git add .
        $ git commit -m "Your detailed description of your changes."
        $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website.

Pull Request Guidelines
^^^^^^^^^^^^^^^^^^^^^^^

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests. Coverage should not go below 70% with `Codecov`.
2. CodeQL should pass.
3. If the pull request adds functionality, the docs should be updated. Add useful documentation to your functionality so it is included in the `docs`.
4. The pull request should work for Python 3.6 through 3.10.
