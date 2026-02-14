package org.bvd.appendix

import com.fasterxml.jackson.databind.ObjectMapper
import java.util.*
import javax.sql.DataSource

class Appendix {
    private val transactionRunner: TransactionRunner
    private val eventStore: EventStore

    constructor(dataSource: DataSource) {
        this.transactionRunner = TransactionRunner(dataSource)
        this.eventStore = EventStore(ObjectMapper().findAndRegisterModules())
        validateConnection(dataSource)
    }

    private fun validateConnection(dataSource: DataSource) {
        dataSource.connection.use { connection ->
            connection.prepareStatement("SELECT version();").use { preparedStatement ->
                preparedStatement.executeQuery().use { resultSet ->
                    while (resultSet.next()) {
                        println(
                            "Appendix initialized with connection: ${dataSource.connection.metaData}, database version: ${
                                resultSet.getString(
                                    1
                                )
                            }"
                        )
                    }
                }
            }
        }
    }

    fun appendEvents(streamId: UUID, events: List<Any>, expectedVersion: Long) {
        transactionRunner.runInTransaction { connection ->
            eventStore.append(connection, events, streamId, expectedVersion)
        }
    }

    fun loadEvents(streamId: UUID): List<Event> {
        return transactionRunner.runInTransaction { connection -> eventStore.loadEvents(connection, streamId) }
    }
}