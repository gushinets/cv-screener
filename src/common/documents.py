# ------------------------------------------------------------------------------
# Supports the document end point
# ------------------------------------------------------------------------------
'''
Contains the implementation of the database, which is actually a fault
tolerant json file to keep track locally of conversations and categories
'''
import os
from .store import DocumentStoreBase, DocFilter
from typing import Dict, Any
from typing import TypedDict, List
from threading import Thread
from uuid import uuid4


class FileStatus(TypedDict):
    status: str
    path: str
    totalSize: int
    currentSize: int
    error: str


class LoadStatus(TypedDict):
    state: int
    current: FileStatus | None
    listSelected: List[str]
    listCompleted: List[FileStatus]
    countSelected: int
    countCompleted: int
    countErrors: int


class DocDir:
    '''
    Privates
    '''
    STATE_NONE = 0
    STATE_PREPARING = 1
    STATE_LOADING = 2
    STATE_STOPPING = 3
    STATE_COMPLETE = 4

    '''
    Keeps track of the documents that are loaded into the database. This
    is mainly for KoRa since the enterprise platform will have too many
    objects to keep in memory
        {
            "C:": {
                "test": {
                    "pdf": {
                        "20.pdf": None
                    }
                }
            }
        }
    '''
    _docNodes: Dict[str, Dict | None] | None = None
    _store: DocumentStoreBase
    _loadThread: Thread | None = None
    _loadStatus: LoadStatus

    def __init__(self, store: DocumentStoreBase):
        '''
        Init this by saving the store we will access
        '''
        self._store = store
        self._loadThread = None
        self._loadStatus = {
            'state': self.STATE_NONE,
            'current': None,
            'listSelected': [],
            'listCompleted': [],
            'countSelected': 0,
            'countCompleted': 0,
            'countErrors': 0,
        }

    def _buildDocNodes(self) -> None:
        '''
        Build a complete tree of all the files in the database
        '''
        # If we already have it build, done
        if self._docNodes is not None:
            return

        # Initialize it
        self._docNodes = {}

        # Set offset and limit
        offset = 0
        limit = 1000

        # Do until we find the end
        offset = 0
        while True:
            # Get the paths
            paths = self._store.getPaths('', offset=offset, limit=limit)

            # For each file
            for path in paths:
                # Get the object id
                objectId = paths[path]

                # Split it into path components
                comps = path.split('/')

                # Now, for each component
                dir: Any = self._docNodes
                for index in range(0, len(comps)):
                    # Get the component name
                    comp = comps[index]

                    # Determine what we should put as the value (dir={}, file=objectId)
                    value: Any
                    if index == len(comps) - 1:
                        value = objectId
                    else:
                        value = {}

                    # If it isn't there, add it
                    if comp not in dir:
                        dir[comp] = value

                    # Move into it
                    dir = dir[comp]

            # Advance the offset
            offset += limit

            # If we did not get a full result, we are done
            if len(paths) < limit:
                break

    def _loadFile(self, loader: Loader, path: str) -> None:
        '''
        Loads a file into the vdb
        '''
        # Preset to no failure
        failure = None

        # Clear the nodes so we rebuild
        self._docNodes = None

        # Create this file status object
        fileStatus: FileStatus = {
            'status': '',
            'error': '',
            'path': path,
            'totalSize': 0,
            'currentSize': 0
        }

        # Set it to none
        objectLoader = None

        try:
            # If it is not already loaded
            if not self.isDocumentLoaded(path):
                # Save the status of this file
                self._loadStatus['current'] = fileStatus

                # Build a url out of it
                url = 'filesys://KoRa/' + path

                # Get a loader object
                objectLoader = loader.getObjectLoader(url)

                # Get the size of the file size
                objectLoader.objectId = str(uuid4())

                objectLoader.size = os.path.getsize(path)
                objectLoader.permissionId = -1

                # Save the size in the loader
                fileStatus['totalSize'] = objectLoader.size

                # Begin the object
                objectLoader.beginObject()

                # Open and enumerate the file...
                limit = 16 * 1024
                offset = 0
                with open(path, 'rb') as file:
                    # While we have read a full buffer
                    while True:
                        # Seek to the proper offset
                        file.seek(offset)

                        # Attempt to read the buffer
                        buffer = file.read(limit)

                        # Get how much we actually read
                        bytesRead = len(buffer)

                        # Write it
                        objectLoader.writeData(buffer)

                        # Update the offset
                        offset = offset + bytesRead

                        # Update the status
                        fileStatus['currentSize'] = fileStatus['currentSize'] + bytesRead

                        # If we did not read a full buffer, end of file
                        if bytesRead != limit:
                            break

        except Exception as e:

            # Save the failure
            failure = e

        try:
            # End this object normally
            if objectLoader is not None:
                objectLoader.endObject()

        except Exception as e:

            # Save the failure
            if failure is None:
                failure = e

        finally:
            # Specifically destroy it
            if objectLoader is not None:
                del objectLoader
                objectLoader = None

        # Save the stats
        if failure is None:
            fileStatus['status'] = 'Completed'
            fileStatus['error'] = ''
            self._loadStatus['listCompleted'].append(fileStatus)
            self._loadStatus['countCompleted'] = self._loadStatus['countCompleted'] + 1
        else:
            fileStatus['status'] = 'Error'
            fileStatus['error'] = str(failure)
            self._loadStatus['listCompleted'].append(fileStatus)
            self._loadStatus['countErrors'] = self._loadStatus['countErrors'] + 1

        # No longer in progress
        self._loadStatus['current'] = None

    def _loadFiles(self) -> None:
        '''
        Thread that runs the loading - this will create a new Loader,
        begin it, load the files and terminate. Files can be added
        to the list during loading
        '''
        try:
            # Set it to none
            loader = None

            # Create a new loader instance
            loader = Loader()

            # Start the loader
            loader.beginLoad()

            # Don't do a forlooop as the list is changing
            while True:
                # If we are supposed to stop, do so now
                if self._loadStatus['state'] == self.STATE_STOPPING:
                    break

                # If we have no more entries, stop
                if len(self._loadStatus['listSelected']) < 1:
                    break

                # Say we are loading
                self._loadStatus['state'] = self.STATE_LOADING

                # Get the path
                path = self._loadStatus['listSelected'].pop(0)

                # Load it
                self._loadFile(loader, path)

        except Exception as e:
            # Output it
            debug('Error in loading files', e)

        # Indicate we are terminating
        self._loadThread = None

        try:
            # Done loading
            if loader is not None:
                loader.endLoad()

        except Exception as e:
            # Output it
            debug('Error in loadFiles', e)

        finally:
            # Specifically destroy it
            if loader is not None:
                del loader
                loader = None

        # Update the status
        self._loadStatus['state'] = self.STATE_COMPLETE

    def isDocumentLoaded(self, path: str) -> bool:
        '''
        Determines if a document is loaded in the store or not (by full path)
        '''
        # We use /, not \\
        path = path.replace('\\', '/')

        # Create a filter
        docFilter = DocFilter({
            "parent": path,
            "chunk": 0
        })

        # Query to get the first chunk
        docs = self._store.get(docFilter=docFilter)

        # If we have a chunk, it is loaded
        if len(docs):
            return True
        else:
            return False

    def addDocument(self, path: str) -> None:
        '''
        This function will add a file to the list, and if the load thread is
        not currently running, start it
        '''
        # We use /, not \\
        path = path.replace('\\', '/')

        # If we do not have a loader
        if not self._loadThread:
            # Setup the stats
            self._loadStatus['state'] = self.STATE_PREPARING
            self._loadStatus['listSelected'] = []
            self._loadStatus['listCompleted'] = []
            self._loadStatus['countSelected'] = 0
            self._loadStatus['countCompleted'] = 0
            self._loadStatus['countErrors'] = 0

            # Create the load thread
            self._loadThread = Thread(group=None, target=self._loadFiles, name='FileLoader')
            self._loadThread.start()

        # Append the file and increase the count
        self._loadStatus['listSelected'].append(path)
        self._loadStatus['countSelected'] = self._loadStatus['countSelected'] + 1

    def addDocuments(self, paths: List[str]) -> None:
        '''
        Add files to load
        '''
        # For each path specified
        for path in paths:
            self.addDocument(path)

    def removeDocument(self, path: str) -> None:
        '''
        Removes a document (by full path or a parent path) from the store
        '''
        # Clear the nodes so we rebuild
        self._docNodes = None

        # We use /, not \\
        path = path.replace('\\', '/')

        # Create a filter
        docFilter = DocFilter({
            "parent": path,
            "chunk": 0,
            "limit": 1000
        })

        while True:
            # Query to get the first chunk
            docs = self._store.get(docFilter=docFilter)

            # If we dont have any chunks, we are done
            if not len(docs):
                break

            # Get the document we found
            objectIds: List[str] = []
            for doc in docs:
                objectId = doc.metadata['objectId']
                objectIds.append(objectId)

            # Remove them
            self._store.remove(objectIds)

        # Clear the nodes so we rebuild
        self._docNodes = None

    def removeDocuments(self, paths: List[str]) -> None:
        '''
        Removes a list of documents
        '''
        # For each path specified
        for path in paths:
            self.removeDocument(path)

    def cancelUpload(self):
        '''
        Cancels all the uploads
        '''
        # If we have a load thread, say to stop
        if self._loadThread:
            self._loadStatus['state'] = self.STATE_STOPPING
        else:
            self._loadStatus['state'] = self.STATE_NONE

    def getStatus(self):
        '''
        Returns the current load status of document loading
        '''
        return self._loadStatus

    def getChildren(self, path: str) -> Dict[str, bool]:
        '''
        Returns the child components of the given path
        '''
        # Build the nodes if we need to
        self._buildDocNodes()

        # Split it into path components
        if path is None or not path:
            comps = []
        else:
            # We use /, not \\
            path = path.replace('\\', '/')

            # Split it
            comps = path.split('/')

        # Build the children
        children: Dict[str, bool] = {}

        # Now, for each component
        dir: Any = self._docNodes
        for index in range(0, len(comps)):
            # Get the component name
            comp = comps[index]

            # If we didn't find it, it is empty
            if comp not in dir:
                return {}

            # Move into it
            dir = dir[comp]

        # If the path specified the complete path and filename,
        # this will be none, so, it has no children
        if isinstance(dir, str):
            return dir

        # Walk through all it's children
        for key in dir:
            if isinstance(dir[key], str):
                # Append the entry
                children[key] = dir[key]
            else:
                children[key] = {}

        # And return it
        return children
